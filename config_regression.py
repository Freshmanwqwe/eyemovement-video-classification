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
    CLIP_STRIDE = 60
    # 视频取3通道
    USE_RGB = False
    # USE_RGB为True时IN_CHANNEL应为3，否则为1
    IN_CHANNEL = 1
    # 回归输出通道为1
    OUT_CHANNEL = 1
    # 数据分辨率
    INPUT_FRAME_SIZE = (400, 200)
    # 提供训练只取NUM_SAMPLES个连续的clips
    RAND_SAMPLE_ENABLE = False
    NUM_SAMPLES = 2
    # 训练用双眼视频
    USE_2EYES = True
    # 降低视频帧率
    FRAME_STEP = 2
    
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
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    # 学习率
    LR = 1e-4
    # epoch
    EPOCHS = 100
    # 余弦退火调度器
    MAX_T = 50
    MIN_LR = 1e-5
    # 定期保存
    SAVE_EPOCHS = 10

    # 保存路径
    OUTPUT_PATH = r"D:\_work\Microsoft\_work\GraduationProject\src\current\output"
    
    
class TimesNetConfig:
    task_name = 'classification'
    seq_len = Config.CLIP_FRAMES
    label_len = 0
    pred_len = 0
    enc_in = 128
    d_model = 64
    d_ff = 128
    e_layers = 2
    top_k = 2
    num_kernels = 3
    embed = 'fixed'
    freq = 'h'
    dropout = 0.1
    c_out = 128
    num_class = 128
    