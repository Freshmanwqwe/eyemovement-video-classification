import torch
import torch.nn as nn
from model.res_model import VideoResnet


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.query_toke = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        q = self.query_toke.repeat(x.size(0), 1, 1)
        attn_output, _ = self.cross_attn(query=q, key=x, value=x)
        x_fusion = self.norm(q + attn_output)
        return x_fusion.squeeze(1)
        

# x = [BATCH, EXP=6, 256], EXP为6，代表6个实验，进入6个输入头
class EyeModel(nn.Module):
    def __init__(self, in_ch, out_ch, filters=[256]):
        super(EyeModel, self).__init__()
        self.exp_models = nn.ModuleList([
            ExpModel(in_ch, filters[0]) for _ in range(6)
        ])
        self.cross_attn = CrossAttentionFusion()
        self.fc = nn.Linear(filters[0], out_ch)
        
        
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
        # x = [BATCH, 2]
        x = self.fc(x)
        
        return x
    
    
class SelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, max_len=100):
        super(SelfAttention, self).__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        attn_out, _ = self.mha(x, x, x)
        x = self.norm(attn_out + x)
        
        return x
        
        
# x = [BATCH, CLIP, 128]
class ExpModel(nn.Module):
    def __init__(self, in_ch, out_ch, filters=[128]):
        super(ExpModel, self).__init__()
        self.encoder = VideoResnet(in_ch, filters[0])
        self.self_attn = SelfAttention()
        self.layer = nn.Sequential(
            nn.Conv1d(filters[0], out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.encoder(x)
        print(f"encoder:{ x.shape }")
        x = self.self_attn(x)
        x = x.transpose(1, 2)
        x = self.layer(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        
        return x
    

if __name__ == "__main__":
    model = EyeModel(1, 2).to('cuda')
    x = [torch.randn(4, 20, 10, 1, 20, 20).to('cuda') for _ in range(6)]
    # print(f"x:{ x.shape }")
    y = model(x)
    print(f"y:{ y.shape }")
