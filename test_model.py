import torch
from model.timesnet3D import TimesNet3D


if __name__ == "__main__":
    model = TimesNet3D(1, 128).to('cuda')
    x = torch.randn(1, 5, 60, 1, 200, 200).to('cuda')
    print(f"x:{x.shape}")
    y = model(x)
    print(f"y:{y.shape}")