import torch
import os
import json
import cv2
import numpy as np
import random

from torchvision.transforms import v2
from torch.utils.data import Dataset
from config_regression import Config

class EyeDataset(Dataset):
    def __init__(self, root_dir, selected_folder_json_path, is_train = True):
        self.root_dir = root_dir
        with open(selected_folder_json_path, 'r') as file:
            self.selected_folder = json.load(file)
        self.patient_nums, self.labels = self._load_patient()
        self.is_train = is_train
        if Config.USE_RGB:
            self.transforms_train = v2.Compose([
                v2.Resize(Config.INPUT_FRAME_SIZE),
                # v2.RandomRotation((-5, 5)),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            self.transforms_val = v2.Compose([
                v2.Resize(Config.INPUT_FRAME_SIZE),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transforms_train = v2.Compose([
                v2.Resize(Config.INPUT_FRAME_SIZE),
                # v2.RandomRotation((-5, 5)),
                v2.Normalize(mean=[0.5], std=[0.5])
            ])
            self.transforms_val = v2.Compose([
                v2.Resize(Config.INPUT_FRAME_SIZE),
                v2.Normalize(mean=[0.5], std=[0.5])
            ])
    
    def __getitem__(self, index):
        patient_num = self.patient_nums[index]
        label = self.labels[index]
        patient_path = os.path.join(self.root_dir, patient_num)
        
        # [EXP, CLIP, T, C, H, W]
        data = []
        
        for experiment in Config.EXPERIMENTS:
            if experiment in self.selected_folder[patient_num]:
                selected_subdir = self.selected_folder[patient_num][experiment]
                video_dir = os.path.join(patient_path, experiment, selected_subdir)
                if Config.USE_2EYES:
                    experiment_clips = self._load_2eyes_video(video_dir, experiment)
                else:
                    experiment_clips = self._load_video(video_dir, experiment)
                data.append(experiment_clips)
            else:
                raise Exception(f'{patient_num} miss the {experiment} experiment')
                    
        return data, torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def _load_patient(self):
        patient_nums = []
        labels = []
        for patient_num in os.listdir(self.root_dir):
            if not os.path.isdir(os.path.join(self.root_dir, patient_num)):
                continue
            patient_path = os.path.join(self.root_dir, patient_num)
            label_path = os.path.join(patient_path, "label.json")
            
            # 跳过没有label.json或label.json里MMSE为'/'的病诊号
            if not os.path.exists(label_path):
                continue
            with open(label_path, 'r') as file:
                label_json = json.load(file)
                if (label_json['MMSE'] == '/'):
                    continue
                
            labels.append(float(label_json['MMSE']))
            patient_nums.append(patient_num)
        return patient_nums, labels
    
    def _load_2eyes_video(self, video_dir, selected_experiment):
        video_cap_l = None
        video_cap_r = None
        for filename in os.listdir(video_dir):
            if filename.endswith('EIL.mp4'):
                video_cap_l = cv2.VideoCapture(os.path.join(video_dir, filename))
            elif filename.endswith('EIR.mp4'):
                video_cap_r = cv2.VideoCapture(os.path.join(video_dir, filename))
        
        clips = []
        if video_cap_l is not None and video_cap_r is not None:
            try:
                clips = self._clip_2eyes_video(video_cap_l, video_cap_r, selected_experiment)
            finally:
                video_cap_l.release()
                video_cap_r.release()
        return clips
        
    def _clip_2eyes_video(self, video_cap_l, video_cap_r, selected_experiment):
        # 取前FRAMES_KEEP个帧
        # [T, C, H, W]
        frames = []
        for idx in range(Config.FRAMES_KEEP[selected_experiment]):
            ret_l, frame_l = video_cap_l.read()
            ret_r, frame_r = video_cap_r.read()
            
            if not ret_l or not ret_r:
                raise Exception(f'Video frames less than {Config.FRAMES_KEEP[selected_experiment]}')
            
            if idx % Config.FRAME_STEP != 0:
                continue
            
            # 横向拼接左眼和右眼帧
            frame = np.concatenate((frame_l, frame_r), axis=1)
            
            if Config.USE_RGB:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frame = torch.from_numpy(frame).permute(2, 0, 1)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                frame = torch.from_numpy(frame).unsqueeze(0)
                
            frames.append(frame)
        
        # 滑动窗口
        # [CLIP, T, C, H, W]
        clips = []
        for i in range(0, len(frames) - Config.CLIP_FRAMES + 1, Config.CLIP_STRIDE):
            frames_list = frames[i : i + Config.CLIP_FRAMES]
            frames_tensor = torch.stack(frames_list)
            if self.is_train:
                frames_tensor = self.transforms_train(frames_tensor)
            else:
                frames_tensor = self.transforms_val(frames_tensor)
            clips.append(frames_tensor)
            
        clips = torch.stack(clips)
        
        if Config.RAND_SAMPLE_ENABLE:
            clips_size = clips.shape[0]
            
            if clips_size <= Config.NUM_SAMPLES:
                return clips
            
            start_idx = random.randint(0, clips_size - Config.NUM_SAMPLES)
            
            return clips[start_idx : start_idx + Config.NUM_SAMPLES]
        
        return clips

    def _load_video(self, video_dir, selected_experiment):
        # [CLIP, T, C, H, W]
        clips = []
        for filename in os.listdir(video_dir):
            if filename.endswith('.mp4'):
                video_path = os.path.join(video_dir, filename)
                video_cap = cv2.VideoCapture(video_path)
                try:
                    clips = self._clip_video(video_cap, selected_experiment)
                finally:
                    video_cap.release()
                return clips
        return clips

    def _clip_video(self, video_cap, selected_experiment):
        # 取前FRAMES_KEEP个帧
        # [T, C, H, W]
        frames = []
        for _ in range(Config.FRAMES_KEEP[selected_experiment]):
            # [C, H, W]
            ret, frame = video_cap.read()
            
            if not ret:
                raise Exception(f'Video frames less than {Config.FRAMES_KEEP[selected_experiment]}')
            
            if Config.USE_RGB:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frame = torch.from_numpy(frame).permute(2, 0, 1)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                frame = torch.from_numpy(frame).unsqueeze(0)
                
            frames.append(frame)
        
        # 滑动窗口
        # [CLIP, T, C, H, W]
        clips = []
        for i in range(0, len(frames) - Config.CLIP_FRAMES + 1, Config.CLIP_STRIDE):
            frames_list = frames[i : i + Config.CLIP_FRAMES]
            frames_tensor = torch.stack(frames_list)
            if self.is_train:
                frames_tensor = self.transforms_train(frames_tensor)
            else:
                frames_tensor = self.transforms_val(frames_tensor)
            clips.append(frames_tensor)
            
        clips = torch.stack(clips)
        
        if Config.RAND_SAMPLE_ENABLE:
            clips_size = clips.shape[0]
            
            if clips_size <= Config.NUM_SAMPLES:
                return clips
            
            start_idx = random.randint(0, clips_size - Config.NUM_SAMPLES)
            
            return clips[start_idx : start_idx + Config.NUM_SAMPLES]
        
        return clips
