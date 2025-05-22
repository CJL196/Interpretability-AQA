
def build_backbone(cfg):
    
    backbone = cfg.backbone
    
    if backbone == "i3d":
        from models.backbone.i3d import I3D
        return I3D().cuda()
    elif backbone == "vivit":
        from models.backbone.vivit import ViViT
        return ViViT().cuda()
    else:
        raise ValueError(f'Unsupported dataset name [{backbone}]')

def build_neck(cfg):
    
    neck = cfg.neck
    
    if neck == "TQN":
        from models.neck.TQN import TQN
        return TQN(1024,cfg.q_number,cfg.query_var,cfg.pe,N=cfg.num_layers,max_len=cfg.max_len).cuda()
    elif neck == "TQT":
        from models.neck.TQT import ActionDecoder
        return ActionDecoder().cuda()
    else:
        raise ValueError(f'Unsupported dataset name [{neck}]')

def build_head(cfg):
    
    # head = cfg.head
    
    # from models.head.evaluator import Evaluator_weighted
    # return Evaluator_weighted().cuda()
    from models.head.gdlt import GDLT
    return GDLT(d_model=1024, nhead=1, dim_feedforward=2048, dropout=0.1, num_decoder_layers=1, n_query=4).cuda()
    
    

        

    