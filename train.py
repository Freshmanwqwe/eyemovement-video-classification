import torch
import torch.nn as nn
import torch.optim as optim
import json

from torch.utils.data import DataLoader, random_split
from torch.amp import autocast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from pathlib import Path

from model.model import EyeModel
from dataset import EyeDataset
from config import Config

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
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training:")
    # 梯度归零
    optimizer.zero_grad()
    
    for exp_videos, labels in pbar:
        # exp_videos是长为6的列表，列表里面是tensor，将列表里面的tensor送进DEIVCE
        exp_videos = [v.to(Config.DEVICE) for v in exp_videos]
        labels = labels.to(Config.DEVICE)
        
        # 自动混合精度
        with autocast(Config.DEVICE):
            outputs = model(exp_videos)
            loss = criterion(outputs, labels)
        
        # 反向传播
        scaler.scale(loss).backward()
        # 更新梯度
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix({ 'loss': loss.item(), 'acc': acc })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating:')
        for exp_videos, labels in pbar:
            exp_videos = [v.to(Config.DEVICE) for v in exp_videos]
            labels = labels.to(Config.DEVICE)
            
            outputs = model(exp_videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({ 'loss': loss.item(), 'acc': acc })
            
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=all_labels, y_pred=all_preds, average='binary'
    )
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, precision, recall, f1, conf_mat
    

def start_train():
    (Path(Config.OUTPUT_PATH) / 'checkpoints').mkdir(parents=True, exist_ok=True)
    # 历史数据记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'lr': [],
    }
    best_val_acc = 0.0
    
    # dataLoader
    train_loader, val_loader = get_data_loader()
    # 模型初始化
    model = EyeModel(Config.IN_CHANNEL, Config.OUT_CHANNEL).to(Config.DEVICE)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
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
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        # 验证
        val_loss, val_acc, precision, recall, f1, conf_mat = validate_epoch(
            model, val_loader, criterion
        )
        # 调度器记录一次
        scheduler.step()
        
        # 记录最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_acc': val_acc,
                'val_f1': f1,
                'val_precision': precision,
                'val_recall': recall,
            }, Path(Config.OUTPUT_PATH) / 'checkpoints' / 'best_model.pt')
            print(f"Saving Best model... val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}")
        
        if (epoch + 1) % Config.SAVE_EPOCHS == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_acc': val_acc if 'val_acc' in locals() else 0,
            }, Path(Config.OUTPUT_PATH) / 'checkpoints' / f'epoch_{epoch + 1}.pt')
            print(f"Saving epoch_{epoch + 1} model... val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}")
        
        # 记录历史数据
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        history['lr'].append(current_lr)
        
        print("SUMMARY:")
        print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")
        print(f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        print(f"precision: {precision:.4f}, recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"confusion_mat:\n{conf_mat}")
        print(f"current_lr: {current_lr:.6f}")
        
    with open(Path(Config.OUTPUT_PATH) / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"\nFinish Training!")
    print(f"best_val_acc: {best_val_acc:.4f}")


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
