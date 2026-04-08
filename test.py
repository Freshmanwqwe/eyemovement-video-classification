import torch
import torch.nn as nn
from tqdm import tqdm
from _output.output1.dataset import EyeDataset
from torch.utils.data import DataLoader
from model.model import EyeModel
from _output.output1.config import Config
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def do_test():
    dataset = EyeDataset(
        root_dir=Config.ROOT_DIR,
        selected_folder_json_path=Config.SELECT_FOLDER_PATH,
        is_train=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )
    model = EyeModel(Config.IN_CHANNEL, Config.OUT_CHANNEL).to(Config.DEVICE)
    
    checkpoint_path = "_output/output1/checkpoints/best_model2.pt"  
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    pbar = tqdm(dataloader, desc="Testing...")
    
    with torch.no_grad():
        for exp_videos, labels in pbar:
            exp_videos = [v.to(Config.DEVICE) for v in exp_videos]
            labels = labels.to(Config.DEVICE)
            outputs = model(exp_videos)
            
            loss_tmp = criterion(outputs, labels)
            running_loss += loss_tmp.item()
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({ 'loss': loss_tmp.item(), 'acc': acc })
    
    avg_loss = running_loss / len(dataloader)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=all_labels, y_pred=all_preds, average='binary'
    )
    conf_mat = confusion_matrix(all_labels, all_preds)
    print("SUMMARY:")
    print(f"Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
    print(f"precision: {precision:.4f}, recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"confusion_mat:\n{conf_mat}")


if __name__=="__main__":
    do_test()
