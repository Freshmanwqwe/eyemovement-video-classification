import torch
import torch.nn as nn
from model_regression.res_model import VideoResnet


class SelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, max_len=100, dropout=0.1, ffn_expansion=4):
        super(SelfAttention, self).__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
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
        seq_len = x.size(1)
        # attention
        x = x + self.pos_embed[:, :seq_len, :]
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(attn_out + x)
        
        # ffn
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
    
    
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.1):
        super(ResidualBlock1D, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv_block(x) + self.conv_skip(x)
        return self.relu(x)
        
        
# x = [BATCH, CLIP, T, C, H, W]
# x = [BATCH, CLIP, 128]
# x = [BATCH, CLIP, 256]
# x = [BATCH, 256]
class ExpModel(nn.Module):
    def __init__(self, in_ch, out_ch, filters=[64, 128, 256], dropout=0.1):
        super(ExpModel, self).__init__()
        self.encoder = VideoResnet(in_ch, filters[0])
        # self.self_attn1 = SelfAttention()
        # self.self_attn2 = SelfAttention()
        # self.self_attn3 = SelfAttention()
        # self.self_attn4 = SelfAttention()
        # self.layer = nn.Sequential(
        #     nn.Conv1d(filters[0], out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(out_ch),
        #     nn.ReLU(),
        # )
        self.layer1 = ResidualBlock1D(filters[0], filters[1])
        self.layer2 = ResidualBlock1D(filters[1], filters[2])
        self.layer3 = ResidualBlock1D(filters[2], filters[2])
        self.layer4 = ResidualBlock1D(filters[2], out_ch)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.encoder(x)
        # x = self.self_attn1(x)
        # x = self.self_attn2(x)
        # x = self.self_attn3(x)
        # x = self.self_attn4(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        
        return x