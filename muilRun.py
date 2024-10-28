from EmbeddingsCal import CLIP4Clip
from dataset import WebVideoDataset
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import os
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np 

def parse():
    parser = argparse.ArgumentParser(description="Video Embeddings Gen")
    parser.add_argument('-bs', '--batch-size', type=int, default=32, help='')
    parser.add_argument('-nw', '--num-workers', type=int, default=8, help='')
    local_rank = int(os.environ["LOCAL_RANK"])
    return parser.parse_args(), local_rank

def main():
    args, local_rank = parse()  # Get local_rank from environment
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    tarPath = '/workspace/Data/video_and_image_datas/processed_videos_wds_datas/second_bilibili_00-30/part1'


    dataset = WebVideoDataset(tarPath)
    sampler = DistributedSampler(dataset, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler
    )

    model_base = CLIP4Clip(model_name='clip_vit_b32',device='cuda:{}'.format(local_rank)) 
    model = DDP(model_base.to(torch.device('cuda:{}'.format(local_rank))), device_ids=[local_rank],output_device=local_rank)

    video_key = []
    video_emb = []
    for batch_idx, (key, keyframes) in enumerate(dataloader):
        Emb_v = model(keyframes, modality='video')

        video_key.extend(key)
        video_emb.extend(Emb_v)
        
    gathered_emb = [None for _ in range(dist.get_world_size())]
    gathered_video_ids = [None for _ in range(dist.get_world_size())]

    dist.all_gather_object(gathered_emb, video_emb)
    dist.all_gather_object(gathered_video_ids, video_key)

    all_emb = [sim for sublist in gathered_emb for sim in sublist]
    all_video_ids = [vid for sublist in gathered_video_ids for vid in sublist]
    if int(os.environ['LOCAL_RANK']) == 0:
        np.save('video_embeddings.npy', np.array(all_emb, dtype=np.float32))
        np.save('video_ids.npy', np.array(all_video_ids))    
