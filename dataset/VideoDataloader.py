import numpy as np
import torch
from typing import List, Union
from torchvision import transforms
from PIL import Image as PILImage
from pathlib import Path
import cv2
import torchvision
import os
from torch.utils.data import Dataset
from PIL import Image as PILImage
from typing import List
from tools.video_decoder import VideoDecoder
import av
import torch.nn.functional as F


def resize_and_crop(image, size=(224, 224)):
    # Resize
    image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    print(image.shape)
    # Center crop
    h, w, _ = image.shape
    start_x = (w - size[1]) // 2
    start_y = (h - size[0]) // 2
    return image[start_y:start_y + size[0], start_x:start_x + size[1]]

def gpu_transform(image):
    # 转换到 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize and center crop
    cropped_image = resize_and_crop(image)

    # 转换为张量并归一化
    tensor = torch.from_numpy(cropped_image).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    return (tensor - mean) / std

class VideoDatasetOrigin(Dataset):
    # 使用Towhee库的视频处理方法，有多余步骤
    def __init__(self, video_paths, tfms=None, max_frames=12):
        self.video_paths = video_paths
        self.video_decoder = VideoDecoder(sample_type="uniform_temporal_subsample", args={'num_samples': max_frames})
        self.max_frames = max_frames
        # self.transform = transform if transform else self.default_transform()
        self.transform = tfms
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = list(self.video_decoder(video_path))  # 获取视频帧列表
        slice_len = len(frames)

        video_tensor = np.zeros((self.max_frames, 3, 224, 224), dtype=np.float32)

        for i, frame in enumerate(frames):
            pil_img = PILImage.fromarray(frame, frame.mode)
            transformed_img = self.transform(pil_img)

            if i < self.max_frames:
                video_tensor[i, ...] = transformed_img.cpu().numpy()
        output = torch.as_tensor(video_tensor).float()
        # print(output.size())
        return output
    
class VideoDataset(Dataset):
    def __init__(self, video_paths, transform=None, max_frames=12):
        self.video_paths = video_paths
        # self.video_decoder = VideoDecoder(sample_type="uniform_temporal_subsample", args={'num_samples': max_frames})
        self.max_frames = max_frames
        self.transform = transform if transform else self.default_transform()
        # self.transform = tfms

    def default_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    def extract_keyframes(self, video_path):
        container = av.open(video_path)
        fps = container.streams.video[0].average_rate  # 获取视频帧率
        total_frames = container.streams.video[0].frames

        interval = total_frames // self.max_frames

        frame_count = 0
        test = []
        i = 0
        frames_tensor = torch.zeros((12, 3,224,224), dtype=torch.float32)
        for frame in container.decode(video=0):
            # print("Frame format:", frame.format.name)
            if frame_count % interval == 0 and i < self.max_frames:
                frame = frame.to_ndarray(format='rgb24')
                # frame = PILImage.fromarray(frame)
                frame = self.transform(frame)
                frames_tensor[i,...] = frame
                i += 1
            frame_count += 1

            if i >= self.max_frames:
                break
        # frames_tensor = torch.stack(frames) 
        return frames_tensor
    def custom_transform(self,tensor):
        tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=True, antialias=True)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        return (tensor - mean) / std
        
    def extract_keyframes_CV2(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = total_frames // self.max_frames

        frame_count = 0
        i = 0
        frames_tensor = torch.zeros((12, 3,224,224), dtype=torch.float32)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0 and i < self.max_frames:
                # 直接使用 NumPy 数组，变换应用在 NumPy 数组上
                # frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                # frame = self.custom_transform(frame)
                frames_tensor[i,...] = self.transform(frame)
                i += 1
                # frames.append(frame)
            frame_count += 1

            if i >= self.max_frames:
                break

        cap.release()
        # frames_tensor = torch.stack(frames) 
        return frames_tensor
    def extract_keyframes_Tensor(self, video_path):
        container = av.open(video_path)
        fps = container.streams.video[0].average_rate  # 获取视频帧率
        total_frames = container.streams.video[0].frames

        interval = total_frames // self.max_frames

        frames = []
        frame_count = 0
        test = []
        i = 0
        frames_tensor = torch.zeros((12, 3,224,224), dtype=torch.float32)
        for frame in container.decode(video=0):
            # print("Frame format:", frame.format.name)
            if frame_count % interval == 0 and i < self.max_frames:
                ndarray = frame.to_ndarray(format='rgb24')
                
                tensor_img = torch.from_numpy(ndarray).permute(2, 0, 1).float() / 255.0
                resized_img = F.interpolate(tensor_img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False,antialias=True)
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

                transformed_img = (resized_img - mean) / std
                # transformed_img = self.transform(tensor_img).to('cuda')
                # frames.append(transformed_img)
                frames_tensor[i,...] = transformed_img
                i += 1
            frame_count += 1

            if i >= self.max_frames:
                break
        # frames_tensor = torch.stack(frames)  # 将多个 frame 组成的 list 转换为 tensor
        # print("Frames tensor size:", frames_tensor.size())  # 检查 tensor 的尺寸
        return frames_tensor
    def extract_keyframes_With_Noresize(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = total_frames // self.max_frames

        frames = []
        frame_count = 0
        i = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0 and i < self.max_frames:
                # 直接使用 NumPy 数组，变换应用在 NumPy 数组上
                # frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                # frame = self.custom_transform(frame)
                frames.append(frame)
                i += 1
                # frames.append(frame)
            frame_count += 1

            if i >= self.max_frames:
                break
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).permute(0,3, 1, 2).float() / 255.0
        frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        cap.release()
        # frames_tensor = torch.stack(frames) 
        return frames
    def __len__(self):
        return len(self.video_paths)
    def __getitem__(self, idx):
        
        video_path = self.video_paths[idx]
        keyframes = self.extract_keyframes(video_path)
        return video_path, keyframes

class VideoDatasetEffiency(Dataset):
    def __init__(self, video_paths, tfms=None, max_frames=12):
        self.video_paths = video_paths
        self.video_decoder = VideoDecoder(sample_type="uniform_temporal_subsample", args={'num_samples': max_frames})
        self.max_frames = max_frames
        # self.transform = transform if transform else self.default_transform()
        self.transform = tfms

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def extract_keyframes(self, video_path):
        container = av.open(video_path)
        fps = container.streams.video[0].average_rate  # 获取视频帧率
        total_frames = container.streams.video[0].frames

        interval = total_frames // self.max_frames

        frames = []
        frame_count = 0
        video_tensor = np.zeros((self.max_frames, 3, 224, 224), dtype=np.float32)

        for frame in container.decode(video=0):
            # print("Frame format:", frame.format.name)
            if frame_count % interval == 0 and len(frames) < self.max_frames:
                ndarray = frame.to_ndarray(format='rgb24')
                video_tensor[frame_count, :] = F.interpolate(torch.from_numpy(np.transpose(ndarray, (2, 0, 1))), size=(224, 224), mode='bilinear', align_corners=False)
                pil_img = PILImage.fromarray(ndarray)
                transformed_img = self.transform(pil_img)
                frames.append(transformed_img)
            frame_count += 1

            if len(frames) >= self.max_frames:
                break
        print(video_tensor.size())
        frames_tensor = torch.stack(frames)  # 将多个 frame 组成的 list 转换为 tensor
        # print("Frames tensor size:", frames_tensor.size())  # 检查 tensor 的尺寸
        return video_tensor
    
    def __len__(self):
        return len(self.video_paths)
    def tensor_transform(self,frames_tensor):
        # 直接对张量进行操作，使用 interpolate 进行 resize
        #antialias
        resized_frames = F.interpolate(frames_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        # 归一化（手动实现 normalize）
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=frames_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=frames_tensor.device).view(1, 3, 1, 1)

        normalized_frames = (resized_frames - mean) / std
        return normalized_frames
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        keyframes = self.extract_keyframes(video_path)
        print(keyframes.size())
        return keyframes