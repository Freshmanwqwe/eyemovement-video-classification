import torch
import torch.nn as nn
from tqdm import tqdm
from dataset_regression import EyeDataset
from torch.utils.data import DataLoader
from model_regression.model import EyeModel
from config_regression import Config


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
    
    checkpoint_path = "output/checkpoints/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterion = nn.MSELoss()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Testing...")
    
    with torch.no_grad():
        for exp_videos, labels in pbar:
            exp_videos = [v.to(Config.DEVICE) for v in exp_videos]
            labels = labels.to(Config.DEVICE)
            outputs = model(exp_videos)
            
            loss_tmp = criterion(outputs, labels)
            running_loss += loss_tmp.item()
            preds = torch.argmax(outputs, dim=1)
            
            pbar.set_postfix({ 'loss': loss_tmp.item() })
    
    avg_loss = running_loss / len(dataloader)

    print("SUMMARY:")
    print(f"Loss: {avg_loss:.4f}")
        

if __name__=="__main__":
    do_test()
