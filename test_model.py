import torch
from model_regression.expModel import ExpModel


if __name__ == "__main__":
    model = ExpModel(1, 128).to('cuda')
    x = torch.randn(1, 5, 60, 1, 200, 200).to('cuda')
    print(f"x:{x.shape}")
    y = model(x)
    print(f"y:{y.shape}")