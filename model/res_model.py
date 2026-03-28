import torch
import torch.nn as nn
        
# x = [BATCH, C, T, H, W]
class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock3D, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv_block(x) + self.conv_skip(x)
        return self.relu(x)
        
        
# x = [BATCH, CLIP, T, C, H, W]
class VideoResnet(nn.Module):
    def __init__(self, in_ch, out_ch, filters=[64, 128, 256, 512]):
        super(VideoResnet, self).__init__()
        self.layer_1 = ResidualBlock3D(in_ch, filters[0], stride=1)
        self.layer_2 = ResidualBlock3D(filters[0], filters[1], stride=2)
        self.layer_3 = ResidualBlock3D(filters[1], filters[2], stride=2)
        self.layer_4 = ResidualBlock3D(filters[2], filters[3], stride=2)
        self.layer_5 = ResidualBlock3D(filters[3], out_ch, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        
    def forward(self, x):
        bs, clips, T, C, H, W = x.shape
        # x = [bs * clips, T, C, H, W]
        x = x.view(bs * clips, T, C, H, W)
        # x = [bs * clips, C, T, H, W]
        x = x.transpose(1, 2)
        
        x = self.layer_1(x)
        print(f"after layer1: {x.shape}")
        x = self.layer_2(x)
        print(f"after layer2: {x.shape}")
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        
        x = self.avgpool(x)
        
        x = x.view(bs * clips, -1)
        x = x.view(bs, clips, -1)
        
        return x
    
if __name__ == "__main__":
    model = VideoResnet(1, 128)
    x = torch.randn(1, 1, 1, 1, 200, 200)
    print(f"x:{ x.shape }")
    y = model(x)
    print(f"y:{ y.shape }")
    