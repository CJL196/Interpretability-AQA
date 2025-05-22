
seed = 999
multi_gpu = "0"

# dataset config (logo, gym, fisv)
dataset_name = "fisv"
# swim_dir = "/mnt/welles/scratch/datasets/condor/backup/logo/Video_result" #"./data/logo/Video_result"
# swim_label = "/mnt/welles/scratch/datasets/condor/backup/logo/LOGO Anno&Split" #"./data/logo/LOGO Anno&Split"
# presave =  "/mnt/welles/scratch/datasets/condor/backup/logo/logo_feats"


fisv_dir = "./fis-v"
label = "TES" #PCS

# dataloader config
subset = 0
bs_train = 32
bs_test = 32
num_workers = 4

# network config (i3d, vivit)
backbone = "vivit"

i3d = dict(
    backbone="I3D",
    neck="",
    evaluator="",
    I3D_ckpt_path="model_rgb.pth" 
)

vivit = dict(
    backbone="ViViT",
)

neck = "TQN"
head = "weighted"
# query number
q_number = 136
# variange for initilize query
query_var = 5
# positional embedding method ["query_pe","query_memory_pe","no"]
pe = "query_pe" 
att_loss = True
dino_loss = True
max_len = 136

num_layers = 1



# training config
split_feats = True
epoch_num = 1000
# load_from = "ckpts/i3d_fisv_999_TES_True_5_query_pe_2.pt"
lr = 1e-4
