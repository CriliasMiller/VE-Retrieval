import webdataset as wds
import torch
from torch.utils.data import Dataset, DataLoader
import av
from torchvision import transforms
import glob
from tarfile import open as tar_open
import io
import decord
from decord import VideoReader
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import numpy as np
import logging

class VideoDataset(Dataset):
    def __init__(self, tar_dir, transform=None, max_frames=12):
        # 获取所有 tar 文件路径
        self.tar_paths = glob.glob(f"{tar_dir}/*.tar")
        self.dataset = (
            wds.WebDataset(self.tar_paths)
            .decode("pil")
            .to_tuple("__key__", "mp4")
        )
        self.max_frames = max_frames
        # self.transform = transform if transform else self.default_transform()
        self.video_size = 224
        self.samples = list(self.dataset)
    def extract_keyframes(self, video_data):
        vr = VideoReader(uri=video_data, height=-1, width=-1)
        ori_vlen = len(vr)
        
        if ori_vlen < self.max_frames:
            indices = np.arange(ori_vlen)
        else:
            indices = np.linspace(0, ori_vlen - 1, num=self.max_frames, dtype=int)

        sampled_frames = vr.get_batch(indices).asnumpy()
        
        frames_tensor = torch.from_numpy(sampled_frames) if type(sampled_frames) is not torch.Tensor else sampled_frames  # (T, H, W, C)

        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        frames_tensor = resize(frames_tensor, (self.video_size,self.video_size), interpolation=InterpolationMode.BICUBIC)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)  # (1, C, 1, 1) 形状
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        return frames_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 直接根据 idx 访问样本
        key, video_data = self.samples[idx]
        keyframes = self.extract_keyframes(io.BytesIO(video_data))
        return key, keyframes
