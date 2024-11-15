import webdataset as wds
import torch
from torch.utils.data import Dataset,IterableDataset, DataLoader
import glob
import io
from decord import VideoReader
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import numpy as np
import os
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torchvision import transforms
import random
    
class IterVideoDataset(IterableDataset):
    def __init__(self, tar_dir, transform=None, max_frames=12):
        self.tar_paths = glob.glob(f"{tar_dir}/*.tar")
        self.url = wds.SimpleShardList(self.tar_paths)
        self.max_frames = max_frames
        self.video_size = 224
        self.pipeline = wds.DataPipeline(
            self.url,
            wds.split_by_node,
            wds.split_by_worker, 
            wds.tarfile_to_samples(),
            wds.decode("pil"),
            wds.to_tuple("__key__", "mp4"),
        )
        self.transform = transform if transform else self.default_transform()
        
    def default_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        
    def extract_keyframes_Origin(self, video_data):
        vr = VideoReader(uri=video_data, height=-1, width=-1)
        ori_vlen = len(vr)

        if ori_vlen < self.max_frames:
            indices = np.arange(ori_vlen)
        else:
            indices = np.linspace(0, ori_vlen - 1, num=self.max_frames, dtype=int)

        frames_tensor = torch.zeros((self.max_frames, 3, 224, 224), dtype=torch.float32)
        for i, index in enumerate(indices):
            frame = vr[index].asnumpy() 
            frame = self.transform(frame)
            frames_tensor[i, ...] = frame 

        return frames_tensor
    def __iter__(self):    
        for key, frames in self.pipeline:
            keyframes = self.extract_keyframes_Origin(io.BytesIO(frames)) 
            yield key, keyframes

            