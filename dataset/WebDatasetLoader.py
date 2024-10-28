import os
import torch
import webdataset as wds
import av
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class WebVideoDataset(Dataset):
    def __init__(self, parent_dir, transform=None, max_frames=12):
        self.video_paths = [
            os.path.join(parent_dir, f) for f in os.listdir(parent_dir)
            if f.endswith('.tar')
        ]
        self.max_frames = max_frames
        self.transform = transform if transform else self.default_transform()

        self.video_data = []
        for tar_path in self.video_paths:
            self.load_videos_from_tar(tar_path)

    def load_videos_from_tar(self, tar_path):
        dataset = (
            wds.WebDataset(tar_path)
            .decode("torch")
            .to_tuple("mp4", "json")  # 假设 tar 包中视频的键是 "mp4"
        )
        
        for video_data, meta in dataset:
            key = os.path.basename(video_data.name).split('.')[0]  # 例如 '010101'
            self.video_data.append((key, video_data))

    def default_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def extract_keyframes(self, video_data):
        container = av.open(video_data)
        total_frames = container.streams.video[0].frames
        interval = total_frames // self.max_frames

        frames_tensor = torch.zeros((self.max_frames, 3, 224, 224), dtype=torch.float32)
        i, frame_count = 0, 0

        for frame in container.decode(video=0):
            if frame_count % interval == 0 and i < self.max_frames:
                frame = frame.to_ndarray(format='rgb24')
                frame = self.transform(frame)
                frames_tensor[i] = frame
                i += 1
            frame_count += 1

            if i >= self.max_frames:
                break

        return frames_tensor

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        key, video_data = self.video_data[idx]
        keyframes = self.extract_keyframes(video_data)
        return key, keyframes
