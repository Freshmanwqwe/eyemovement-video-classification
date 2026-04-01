import torch
import torch.nn as nn
from model_regression.expModel import ExpModel


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_exp=6, dropout=0.1, ffn_expansion=4):
        super(CrossAttentionFusion, self).__init__()
        
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.exp_embed = nn.Parameter(torch.randn(1, num_exp, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        ffn_dim = embed_dim * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x + self.exp_embed
        q = self.query_token.expand(batch_size, -1, -1)
        
        # Attention
        attn_output, _ = self.cross_attn(query=q, key=x, value=x)
        x_fusion = self.norm1(q + self.attn_dropout(attn_output))
        
        # FFN
        ffn_output = self.ffn(x_fusion)
        x_fusion = self.norm2(x_fusion + ffn_output)
        
        return x_fusion.squeeze(1)
        

# x = [BATCH, EXP=6, 256], EXP为6，代表6个实验，进入6个输入头
class EyeModel(nn.Module):
    def __init__(self, in_ch, out_ch, filters=[256], dropout=0.1):
        super(EyeModel, self).__init__()
        self.exp_models = nn.ModuleList([
            ExpModel(in_ch, filters[0]) for _ in range(6)
        ])
        self.cross_attn = CrossAttentionFusion()
        self.fc = nn.Linear(filters[0], out_ch)
        self.dropout = nn.Dropout(p=dropout)
        
        
    def forward(self, x_list):
        exp_features = []
        for i, model in enumerate(self.exp_models):
            out = model(x_list[i])
            exp_features.append(out)
        
        # x = [6, BATCH, 256]
        x = torch.stack(exp_features)
        # x = [BATCH, 6, 256]
        x = x.transpose(0, 1)
        # x = [BATCH, 256]
        x = self.cross_attn(x)
        
        x = self.dropout(x)
        
        # x = [BATCH, 1]
        x = self.fc(x)
        
        # sigmoid约束
        x = 30.0 * torch.sigmoid(x)
        
        return x
    

if __name__ == "__main__":
    model = EyeModel(1, 2).to('cuda')
    x = [torch.randn(4, 20, 60, 1, 20, 20).to('cuda') for _ in range(6)]
    # print(f"x:{ x.shape }")
    y = model(x)
    print(f"y:{ y.shape }")
