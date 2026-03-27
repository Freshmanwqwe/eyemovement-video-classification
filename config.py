import os
from pathlib import Path
import torch

class Config:
    EXPERIMENTS = ['AntiSaccade', 'Explore', 'LateralFixation', 'ProSaccade', 'Smooth-Horizontal', 'Smooth-Vertical']
    # 每个视频取前FRAMES_KEEP帧
    FRAMES_KEEP = {
        "AntiSaccade": 1500,
        "Explore": 1500,
        "LateralFixation": 840,
        "ProSaccade": 1500,
        "Smooth-Horizontal": 420,
        "Smooth-Vertical": 420
    }
    # 滑动窗口大小
    CLIP_FRAMES = 60
    # 滑动窗口步数
    CLIP_STRIDE = 30
    # 视频取3通道
    USE_RGB = True
    # USE_RGB为True时IN_CHANNEL应为3，否则为1
    IN_CHANNEL = 3
    # 二分类输出通道为2
    OUT_CHANNEL = 2
    # 数据分辨率
    INPUT_FRAME_SIZE = (200, 200)
    
    # 数据集相关
    ROOT_DIR = r"D:\_work\Programming\Datasets\eyemovement\PG-Videos"
    SELECT_FOLDER_PATH = r"D:\_work\Microsoft\_work\GraduationProject\src\selected_folder.json"
    # 训练集、验证集、测试集比例
    RATIO = {
        'train': 0.7,
        'val': 0.3,
    }
    
    # 训练相关
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # DEVICE = 'cpu'
    # dataLoader
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    # 学习率
    LR = 1e-4
    # epoch
    EPOCHS = 100
    # 余弦退火调度器
    MAX_T = 100
    MIN_LR = 1e-5
    # 定期保存
    SAVE_EPOCHS = 10

    # 保存路径
    OUTPUT_PATH = r"D:\_work\Microsoft\_work\GraduationProject\src\current\output"
    