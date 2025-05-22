import torch
import torch.nn as nn
from models.head.transformer import TransformerDecoderLayer, TransformerDecoder

class GDLT(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_decoder_layers, n_query):
        super(GDLT, self).__init__()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    "relu", layer_norm_eps=1e-5, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.prototype = nn.Embedding(n_query, d_model)

        self.weight = torch.linspace(0, 1, n_query, requires_grad=False).cuda()
        print(self.weight)
        self.regressor = nn.Linear(d_model, n_query)

    def forward(self, x):
        # x (b, t, c)
        b, t, c = x.shape

        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        
        q1 = self.decoder(q, x)
        s = self.regressor(q1)  # (b, n, n)
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n) 提取张量s的对角线元素
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)

        return {'output': out, 'embed': q1}

def get_gdlt_loss(pred, label, feat):
    return LossFun(alpha=1.0, margin=1.0)(pred, label, feat)

# 各种分布特征聚合
class LossFun(nn.Module):
    def __init__(self, alpha, margin):
        super(LossFun, self).__init__()
        self.mse_loss = nn.MSELoss()
        # self.mse_loss = nn.L1Loss()
        from models.head.triplet_loss import HardTripletLoss
        self.triplet_loss = HardTripletLoss(margin=margin, hardest=True)
        self.alpha = alpha

    def forward(self, pred, label, feat):
        # feat (b, n, c), x (b, t, c)
        if feat is not None:
            device = feat.device
            b, n, c = feat.shape
            flat_feat = feat.view(-1, c)  # (bn, c)
            la = torch.arange(n, device=device).repeat(b)

            t_loss = self.triplet_loss(flat_feat, la)
            # t_loss = pair_diversity_loss(feat)
        else:
            self.alpha = 0
            t_loss = 0

        mse_loss = self.mse_loss(pred, label)
        # mse_loss = pearson_loss(pred, label)
        return mse_loss + self.alpha * t_loss, mse_loss, t_loss
        # return f_loss, mse_loss, t_loss, f_loss
