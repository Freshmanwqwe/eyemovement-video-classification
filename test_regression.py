import torch
import torch.nn as nn
from tqdm import tqdm
from dataset_regression import EyeDataset
from torch.utils.data import DataLoader
from model_regression.model import EyeModel
from config_regression import Config
import csv


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
    
    checkpoint_path = "output_regression/checkpoints/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterion = nn.HuberLoss(reduction='none')
    
    pbar = tqdm(dataloader, desc="Testing...")
    
    results = []
    total_loss = 0.0
    total_diff = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for exp_videos, labels in pbar:
            exp_videos = [v.to(Config.DEVICE) for v in exp_videos]
            labels = labels.to(Config.DEVICE, dtype=torch.float32).view(-1, 1)
            outputs = model(exp_videos)
            
            loss_batch = criterion(outputs, labels)
            diff_batch = torch.abs(labels - outputs)
            
            for i in range(labels.size(0)):
                true_label = labels[i].item()
                pred_label = outputs[i].item()
                diff = diff_batch[i].item()
                loss_val = loss_batch[i].item()
                
                results.append({
                    'True Label': true_label,
                    'Predicted Label': pred_label,
                    'Difference': diff,
                    'Loss': loss_val
                })
                
                total_loss += loss_val
                total_diff += diff
                total_samples += 1
            
            avg_loss_batch = loss_batch.mean().item()
            pbar.set_postfix({ 'loss': avg_loss_batch })
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_diff = total_diff / total_samples if total_samples > 0 else 0

    print("SUMMARY:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Difference: {avg_diff:.4f}")
    
    csv_file = "output_regression/output_test.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['True Label', 'Predicted Label', 'Difference', 'Loss'])
        for r in results:
            writer.writerow([r['True Label'], r['Predicted Label'], r['Difference'], r['Loss']])
        writer.writerow([])
        writer.writerow(['Average Loss', 'Average Difference'])
        writer.writerow([avg_loss, avg_diff])
        
    print(f"Results saved to {csv_file}")
        

if __name__=="__main__":
    do_test()
