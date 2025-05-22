
import torch
from tqdm import tqdm
import logging
from utils.utils import log_and_print
from utils.vis import *
from loss import attention_loss, cal_spearmanr_rl2
from torch.utils.tensorboard import SummaryWriter
from models.head.gdlt import get_gdlt_loss
# from methods.weight_methods import NashMTL
import matplotlib.pyplot as plt
def run(cfg, base_logger, network, data_loaders, kld, mse, optimizer, scheduler,splits=["train","test"]):
    backbone, neck, head = network
    logger_exp = logging.getLogger("experiment_logger")
    logger_exp.setLevel(logging.INFO)
    exp_handler = logging.FileHandler("experiment.log")  # 日志文件
    exp_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    exp_handler.setFormatter(exp_formatter)
    logger_exp.addHandler(exp_handler)
    
    rho_best, epoch_best, rl2_best = 0, 0, 0
    test_srcc = []
    
    test_only = "train" not in splits
    if test_only:
        cfg.epoch_num = 400

    writer = SummaryWriter(log_dir='runs/0521-pcs')
    
    for epoch in range(cfg.epoch_num):
        for split in splits:
            true_scores = []
            pred_scores = []
            
            if split == 'train':
                backbone.train()
                head.train()
                neck.train()
                torch.set_grad_enabled(True)
            else:
                backbone.eval()
                head.eval()
                neck.eval()
                torch.set_grad_enabled(False)
                
            self_map_lst = []
            cross_map_lst = []
            losses = 0
            for data_ in data_loaders[split]:
                data, clip_info = data_
                # print(f'clip_info: {clip_info}')
                score = data["score"].float().cuda()
                video = data["video"]
                if cfg.split_feats:
                    
                    
                    if "feats" in data:
                        clip_feats = data["feats"].cuda()  # (B, L, D)
                    else:
                        bs, frame, feats  = video.shape
                        video = video.reshape(video.shape[0],3,48,16,224,224).cuda()
                        clip_feats = torch.empty(bs, video.shape[2], feats).cuda()
                        for i in range(frame):
                            clip_feats[:,i] = backbone(video[:,:,i,:,:,:].cuda())[1].squeeze(-1).squeeze(-1).squeeze(-1)
                else:
                    bs, frame, h, w  = video.shape
                    clip_feats = backbone(video)[1].squeeze(-1).squeeze(-1).permute(0,2,1)
                
                tgt_weight, graph_attn = neck(clip_feats,train=False) # tgt_weight.shape = (B, L, D)
                # probs, weight, means, var = head(tgt_weight) # probs.shape = (B,)
                out = head(tgt_weight)
                probs = out['output']
                loss_head, mse, triplet = get_gdlt_loss(probs, score, out['embed'])
                
                pred_scores.extend([i.item() for i in probs])
                true_scores.extend(score.cpu().numpy())
                
                kld_loss, self_map_lst, cross_map_lst = attention_loss(graph_attn, kld, self_map_lst, cross_map_lst)
                # self_map_vis(graph_attn)
                # if epoch > 5000 and test_only:
                #     user_study(means,weight,clip_info,cfg.dataset_name)
             
                
                
                # if cfg.dino_loss:
                #     probs = probs + torch.randn_like(probs).normal_(mean=0.0, std=0.05) #
                # mes_loss = mse(probs,score)
                if cfg.att_loss:
                    loss = loss_head + kld_loss
                else:
                    loss = loss_head
          
                losses += loss
                rho, p, rl2 = cal_spearmanr_rl2(pred_scores, true_scores)
                
                
                if split=="train":
                    scheduler.step()                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                    
            # show information
            log_and_print(base_logger, f'Epoch: {epoch}, {split} correlation: {rho}, best: {rho_best}')
            writer.add_scalar(f'{split}/correlation', rho, epoch)
            
            if rho > rho_best and split == "test":
                rho_best = rho
                epoch_best = epoch
                rl2_best = rl2
                log_and_print(base_logger, '-----New best found!-----')
                if not test_only:
                    torch.save({'epoch': epoch,
                                'backbone': backbone.state_dict(),
                                'neck': neck.state_dict(),
                                'head': head.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'rho_best': rho_best}, f'ckpts/i3d_{cfg.dataset_name}_{cfg.seed}_{cfg.label}_{cfg.att_loss}_{cfg.query_var}_{cfg.pe}_{cfg.num_layers}.pt') 
                
                

    if test_only:                
        logger_exp.info(f"test dataset: {cfg.dataset_name}, seed: {cfg.seed}, label: {cfg.label}, query_var: {cfg.query_var}, pe: {cfg.pe}, att_loss: {cfg.att_loss}, dino_loss:{cfg.dino_loss}, num_layers: {cfg.num_layers}, SRCC: {rho_best}, RL2: {rl2_best}")
    # logger_exp.info(f"dataset: {cfg.dataset_name}, seed: {cfg.seed}, label: {cfg.label}, query_var: {cfg.query_var}, pe: {cfg.pe}, att_loss: {cfg.att_loss}, dino_loss:{cfg.dino_loss}, num_layers: {cfg.num_layers}, SRCC: {rho_best}, RL2: {rl2_best}")
            