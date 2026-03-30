import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from model.seriesLib.models.TimesNet import Model
from config import TimesNetConfig


# x = [BATCH, CLIP, T, C, H, W]
class TimesNet3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TimesNet3D, self).__init__()
        self.res2d = Resnet2D(in_ch, 128)
        config = TimesNetConfig()
        config.num_class = out_ch
        self.timesnet1D = Model(config)
    
    def forward(self, x):
        # x = [BATCH, CLIP, T, out_ch]
        x = self.res2d(x)
        bs, clips, T, F = x.shape
        # x = [BATCH * CLIP, T, out_ch]
        x = x.view(bs * clips, T, F)
        x_mark = torch.ones(bs * clips, T, device=x.device)
        # x = [BATCH * CLIP, out_ch]
        x = self.timesnet1D(x, x_mark, None, None)
        # x = [BATCH, CLIP, out_ch]
        x = x.view(bs, clips, -1)
        
        return x


class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock2D, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv_block(x) + self.conv_skip(x)
        return self.relu(x)


# x = [BATCH, CLIP, T, C, H, W]
class Resnet2D(nn.Module):
    def __init__(self, in_ch, out_ch, filters=[16, 32, 64, 128]):
        super(Resnet2D, self).__init__()
        self.layer_1 = ResidualBlock2D(in_ch, filters[0], stride=1)
        self.layer_2 = ResidualBlock2D(filters[0], filters[1], stride=2)
        self.layer_3 = ResidualBlock2D(filters[1], filters[2], stride=2)
        self.layer_4 = ResidualBlock2D(filters[2], filters[3], stride=2)
        self.layer_5 = ResidualBlock2D(filters[3], out_ch, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        bs, clips, T, C, H, W = x.shape
        # x = [clips, bs, T, C, H, W]
        x = x.transpose(0, 1).contiguous()
        
        def run_features(inputs):
            out = self.layer_1(inputs)
            out = self.layer_2(out)
            out = self.layer_3(out)
            out = self.layer_4(out)
            out = self.layer_5(out)
            out = self.avgpool(out)
            return out
        
        res = []
        for tmpx in x:
            # tmpx = [bs, T, C, H, W]
            tmpx = tmpx.view(bs * T, C, H, W)
            tmpx = checkpoint.checkpoint(run_features, tmpx, use_reentrant=False)
            tmpx = tmpx.view(bs, T, -1)
            res.append(tmpx)
        
        # [clips, bs, T, 128]
        x = torch.stack(res, dim=0)
        # [bs, clips, T, 128]
        x = x.transpose(0, 1).contiguous()
        
        return x
        
    def _forward(self, x):
        bs, clips, T, C, H, W = x.shape
        # x = [bs * clips * T, C, H, W]
        x = x.view(bs * clips * T, C, H, W)
        
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        
        x = self.avgpool(x)
        
        # x = [bs * clips * T, ot_ch]
        x = x.view(bs * clips * T, -1)
        # x = [bs, clips, T, out_ch]
        x = x.view(bs, clips, T, -1)
        
        return x
    

if __name__ == "__main__":
    model = TimesNet3D(1, 128).to('cuda')
    x = torch.randn(1, 5, 60, 1, 200, 200).to('cuda')
    print(f"x:{x.shape}")
    y = model(x)
    print(f"y:{y.shape}")
