import torch
import torch.nn as nn
import torch.optim as optim
import json

from torch.utils.data import DataLoader, random_split
from torch.amp import autocast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from pathlib import Path

from model_regression.model import EyeModel
from dataset_regression import EyeDataset
from config_regression import Config

def get_data_loader():
    dataset = EyeDataset(
        root_dir=Config.ROOT_DIR,
        selected_folder_json_path=Config.SELECT_FOLDER_PATH,
    )
    if len(dataset) == 0:
        raise Exception("数据集为空")
    
    # 划分数据集
    val_size = int(len(dataset) * Config.RATIO['val'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # dataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def train_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    running_dif = 0.0
    min_train_loss = float('inf')
    max_train_loss = 0.0
    min_train_dif = float('inf')
    max_train_dif = 0
    
    pbar = tqdm(dataloader, desc="Training:")
    # 梯度归零
    optimizer.zero_grad()
    
    for exp_videos, label in pbar:
        # exp_videos是长为6的列表，列表里面是tensor，将列表里面的tensor送进DEIVCE
        exp_videos = [v.to(Config.DEVICE, non_blocking=True) for v in exp_videos]
        label = label.to(Config.DEVICE, dtype=torch.float32, non_blocking=True).view(-1, 1)
        
        # 自动混合精度
        with autocast(Config.DEVICE):
            outputs = model(exp_videos)
            loss = criterion(outputs, label)
        
        # 反向传播
        scaler.scale(loss).backward()
        # 更新梯度
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        max_train_loss = max(max_train_loss, loss.item())
        min_train_loss = min(min_train_loss, loss.item())
        
        with torch.no_grad():
            outputs_clamped = torch.clamp(outputs, 0.0, 30.0)
            diff = torch.abs(label - outputs_clamped)
            
            batch_dif_sum = torch.sum(diff).item()
            running_dif += batch_dif_sum
            
            max_train_dif = max(max_train_dif, diff.max().item())
            min_train_dif = min(min_train_dif, diff.min().item())
        
        running_loss += loss.item() * label.size(0)
        
        pbar.set_postfix({ 'loss': loss.item(), 'avg_dif': batch_dif_sum / label.size(0),
                          'max_dif': diff.max().item() })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    avg_train_dif = running_dif / len(dataloader.dataset)
    
    return epoch_loss, min_train_loss, max_train_loss, min_train_dif, max_train_dif, avg_train_dif


def validate_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_dif = 0.0
    min_val_loss = float('inf')
    max_val_loss = 0.0
    min_val_dif = float('inf')
    max_val_dif = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating:')
        for exp_videos, label in pbar:
            exp_videos = [v.to(Config.DEVICE, non_blocking=True) for v in exp_videos]
            label = label.to(Config.DEVICE, dtype=torch.float32, non_blocking=True).view(-1, 1)
            
            with autocast(Config.DEVICE):
                outputs = model(exp_videos)
                loss = criterion(outputs, label)
            
            max_val_loss = max(max_val_loss, loss.item())
            min_val_loss = min(min_val_loss, loss.item())
            
            outputs_clamped = torch.clamp(outputs, 0.0, 30.0)
            diff = torch.abs(label - outputs_clamped)
            
            batch_dif_sum = torch.sum(diff).item()
            running_dif += batch_dif_sum
            
            max_val_dif = max(max_val_dif, diff.max().item())
            min_val_dif = min(min_val_dif, diff.min().item())
            
            running_loss += loss.item() * label.size(0)
            
            pbar.set_postfix({ 'loss': loss.item(), 'avg_dif': batch_dif_sum / label.size(0),
                          'max_dif': diff.max().item() })
            
    epoch_loss = running_loss / len(dataloader.dataset)
    avg_val_dif = running_dif / len(dataloader.dataset)
    
    return epoch_loss, min_val_loss, max_val_loss, min_val_dif, max_val_dif, avg_val_dif
        

def start_train():
    (Path(Config.OUTPUT_PATH) / 'checkpoints').mkdir(parents=True, exist_ok=True)
    # 历史数据记录
    history = {
        'train_loss': [],
        'min_train_loss': [],
        'max_train_loss': [],
        'min_train_dif': [],
        'max_train_dif': [],
        'avg_train_dif': [],
        'val_loss': [],
        'min_val_loss': [],
        'max_val_loss': [],
        'min_val_dif': [],
        'max_val_dif': [],
        'avg_val_dif': [],
        'lr': [],
    }
    best_val_loss = float('inf')
    
    # dataLoader
    train_loader, val_loader = get_data_loader()
    # 模型初始化
    model = EyeModel(Config.IN_CHANNEL, Config.OUT_CHANNEL).to(Config.DEVICE)
    # 既然目标分数的尺度变成了 [0, 30]，HuberLoss 的 delta 相应的需要放大
    # 原来是 2.0/30=0.06，现在就设为 2.0，即误差在2分以内容忍，超过2分按照L1严厉算
    criterion = nn.HuberLoss(delta=2.0)
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    # 调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, Config.MAX_T, Config.MIN_LR)
    # 混合精度
    scaler = torch.cuda.amp.GradScaler()
    
    # 训练循环
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1} / { Config.EPOCHS }")
        print("-" * 50)
        
        # 训练
        train_loss, min_train_loss, max_train_loss, min_train_dif, max_train_dif, avg_train_dif = train_epoch(model, train_loader, criterion, optimizer, scaler)
        # 验证
        val_loss, min_val_loss, max_val_loss, min_val_dif, max_val_dif, avg_val_dif = validate_epoch(model, val_loader, criterion)
        # 调度器记录一次
        scheduler.step()
        
        # 记录最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, Path(Config.OUTPUT_PATH) / 'checkpoints' / 'best_model.pt')
            print(f"Saving Best model... val_loss: {val_loss:.4f}")
        
        if (epoch + 1) % Config.SAVE_EPOCHS == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, Path(Config.OUTPUT_PATH) / 'checkpoints' / f'epoch_{epoch + 1}.pt')
            print(f"Saving epoch_{epoch + 1} model... val_loss: {val_loss:.4f}")
        
        # 记录历史数据
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['min_train_loss'].append(min_train_loss)
        history['max_train_loss'].append(max_train_loss)
        history['min_train_dif'].append(min_train_dif)
        history['max_train_dif'].append(max_train_dif)
        history['avg_train_dif'].append(avg_train_dif)
        history['val_loss'].append(val_loss)
        history['min_val_loss'].append(min_val_loss)
        history['max_val_loss'].append(max_val_loss)
        history['min_val_dif'].append(min_val_dif)
        history['max_val_dif'].append(max_val_dif)
        history['avg_val_dif'].append(avg_val_dif)
        history['lr'].append(current_lr)
        
        print("SUMMARY:")
        print(f"train_loss: {train_loss:.4f}; min_train_loss: {min_train_loss:.4f}; max_train_loss: {max_train_loss:.4f}; min_train_dif: {min_train_dif:.4f}; max_train_dif: {max_train_dif:.4f}; avg_train_dif: {avg_train_dif:.4f}")
        print(f"val_loss: {val_loss:.4f}; min_val_loss: {min_val_loss:.4f}; max_val_loss: {max_val_loss:.4f}; min_val_dif: {min_val_dif:.4f}; max_val_dif: {max_val_dif:.4f}; avg_val_dif: {avg_val_dif:.4f}")
        print(f"current_lr: {current_lr:.6f}")
        
    with open(Path(Config.OUTPUT_PATH) / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"\nFinish Training!")
    print(f"best_val_loss: {best_val_loss:.4f}")


def test_dataset():
    train_loader, val_loader = get_data_loader()
    batch_data = next(iter(train_loader))[0]
    print(type(batch_data))
    print(type(batch_data[0]))
    print(batch_data[0].shape)
    
def test_model():
    from torchinfo import summary
    model = EyeModel(Config.IN_CHANNEL, Config.OUT_CHANNEL).to(Config.DEVICE)
    input_list = [
        torch.randn(Config.BATCH_SIZE, 1, 1, 3, 1, 1).to(Config.DEVICE)
        for _ in range(6)
    ]
    
    summary(
        model,
        input_data=[input_list],
        col_names=("input_size", "output_size", "num_params", "mult_adds"), 
        depth=4
    )
    
    
if __name__ == "__main__":
    start_train()
    # test_model()
