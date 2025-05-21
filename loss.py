import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats


def attention_loss(graph_attn, kld, self_map_lst, cross_map_lst):
    kld_loss = 0
    self_maps, cross_maps, memorys = graph_attn
    for self_map, cross_map, memory in zip(self_maps, cross_maps, memorys):
        # self attention map
        G_D = torch.matmul(self_map.transpose(0,1), self_map.transpose(0,1).transpose(-1,-2))
        G_D = torch.div(G_D,self_map.size(-1))
        soft_gd_self = F.softmax(G_D,dim=1)

        # cross attention map
        G_E = torch.matmul(cross_map.transpose(0,1), cross_map.transpose(0,1).transpose(-1,-2))
        G_E = torch.div(G_E,cross_map.size(-1))
        soft_gd_cross = F.softmax(G_E,dim=1)
        # show_heatmaps(G_D[0].cpu(), 'self', 'self', name = f'self_attn_map')
        # show_heatmaps(G_E[0].cpu(), 'cross', 'cross', name = f'cross_attn_map')
        kld_loss +=  kld(F.log_softmax(G_D, dim=-1), F.softmax(G_E, dim=-1))

    for i in range(len(soft_gd_cross)):
        self_map_lst.append(soft_gd_self[i])
        cross_map_lst.append(soft_gd_cross[i])
        
    return kld_loss, self_map_lst, cross_map_lst

def cal_spearmanr_rl2(pred_scores, true_scores):
    # calculate spearman correlation
    rho, p = stats.spearmanr(pred_scores, true_scores)
    # calculate rl2
    pred_scores_ = np.array(pred_scores)
    true_scores_ = np.array(true_scores)
    rl2 = np.power((pred_scores_ - true_scores_) / (true_scores_.max() - true_scores_.min()), 2).sum() / true_scores_.shape[0]
    return rho, p, rl2

from matplotlib import pyplot as plt
def show_heatmaps(matrices, xlabel, ylabel, titles=None,
                  cmap='Reds', name = 'attention_heatmap'):
    """显示矩阵热图"""
    num_rows, num_cols = matrices.shape
    # 调整图形大小，增加高度
    fig, axes = plt.subplots(1, 1, figsize=(30, 15))
    
    # 处理单个热图的情况
    im = axes.imshow(matrices.detach().numpy(), cmap=cmap, aspect='auto')
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    if titles:
        axes.set_title(titles)
    
    # 添加颜色条
    fig.colorbar(im, ax=axes)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f'{name}.png')
    plt.close()