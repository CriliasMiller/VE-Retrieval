
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import time
import argparse
import torch.multiprocessing as mp
from dataset.video_dataloader import VideoDataset
from towhee.models.clip4clip import convert_tokens_to_id
from towhee.models import clip4clip
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP

class CLIP4Clip(nn.Module): 
    def __init__(self, model_name: str, modality: str = None, weight_path: str = None, device: str = None):
        nn.Module.__init__(self)  
        super().__init__()
        
        self.modality = modality
        if weight_path is None:
            weight_path = '../clip4clip/pytorch_model.bin.1'
        self.device = device
        self.model = clip4clip.create_model(model_name=model_name,
                                            context_length=77,
                                            pretrained=True,
                                            weights_path=weight_path,
                                            device=self.device)

        self.tokenize = clip4clip.SimpleTokenizer()
        self.tfms = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.model.eval()

    def forward(self, data, modality):
        if modality == 'video':
            vec = self._inference_from_video(data)
        elif modality == 'text':
            vec = self._inference_from_text(data)
        else:
            raise ValueError("modality[{}] not implemented.".format(modality))
        return vec

    def _inference_from_text(self, text: str):
        self.model.eval()
        text_ids = convert_tokens_to_id(self.tokenize, text)
        text_ids = torch.tensor(text_ids).unsqueeze(0).to(self.device)
        text_features = self.model.module.get_sequence_output(text_ids) if isinstance(self.model, nn.DataParallel) else self.model.get_sequence_output(text_ids)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach().flatten().cpu().numpy()

    def _inference_from_video(self, video_input):
        self.model.eval()
        video = video_input.to(self.device)
    
        b, ts, channel, h, w = video.shape
        video = video.view(b*ts, channel, h, w)
        video_mask = np.zeros((b, ts), dtype=np.int32)
        video_mask[: , :ts] = [1] * ts
        video_mask = torch.as_tensor(video_mask).float().to(self.device)

        visual_output = self.model.get_visual_output(video, video_mask, shaped=True)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        visual_output = torch.sum(visual_output, dim=1) / video_mask_un_sum
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        return visual_output.detach().cpu().numpy()
    

def parse():
    local_rank = int(os.environ["LOCAL_RANK"])
    return local_rank

def originData(local_rank, video_paths,text_emb,batch_size, num_workers):

    model_base = CLIP4Clip(model_name='clip_vit_b32',device='cuda:{}'.format(local_rank)) 
    model = DDP(model_base.to(torch.device('cuda:{}'.format(local_rank))), device_ids=[local_rank],output_device=local_rank)

    # 创建 DataLoader
    dataset = VideoDataset(video_paths)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=sampler)

    print("init dataloader")
    
    logit_scale = model_base.model.clip.logit_scale.exp().detach().cpu().numpy()
    
    video_similarities = []
    video_ids = []

    start_time = time.time()
    for batch_frames in dataloader:
        video_paths, frames = batch_frames
        f = model(frames, modality='video')
        sim = logit_scale * np.matmul(f, text_emb)

        video_similarities.extend(sim.flatten().tolist())
        video_ids.extend(video_paths)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"运行时间: {elapsed_time:.4f} 秒.")
    gathered_similarities = [None for _ in range(dist.get_world_size())]
    gathered_video_ids = [None for _ in range(dist.get_world_size())]

    dist.all_gather_object(gathered_similarities, video_similarities)
    dist.all_gather_object(gathered_video_ids, video_ids)

    all_similarities = [sim for sublist in gathered_similarities for sim in sublist]
    all_video_ids = [vid for sublist in gathered_video_ids for vid in sublist]
    # print(f"Rank {os.environ['LOCAL_RANK']} : {len(all_similarities)}.")
    top_indices = np.argsort(all_similarities)[::-1]
    if int(os.environ['LOCAL_RANK']) == 0:
        for idx in top_indices:
            if all_similarities[idx] >= 25:
                print(f"Video Path: {all_video_ids[idx]}, Similarity: {all_similarities[idx]}")

def cls_sim(all_similarities, all_video_ids):
     if int(os.environ['LOCAL_RANK']) == 0:
        top_indices = np.argsort(all_similarities)[::-1]
        video_list_25 = []
        video_list_24 = []
        video_list_23 = []
        video_list_22 = []
        video_list_21 = []
        video_list_20 = []
        video_list_19 = []
        for idx in top_indices:
            if all_similarities[idx] >= 25:
                print(f"Video Path: {all_video_ids[idx]}, Similarity: {all_similarities[idx]}")
                video_list_25.append(all_video_ids[idx])
            elif all_similarities[idx] >= 24:
                print(f"Video Path: {all_video_ids[idx]}, Similarity: {all_similarities[idx]}")
                video_list_24.append(all_video_ids[idx])
            elif all_similarities[idx] >= 23:
                print(f"Video Path: {all_video_ids[idx]}, Similarity: {all_similarities[idx]}")
                video_list_23.append(all_video_ids[idx])
            elif all_similarities[idx] >= 22:
                print(f"Video Path: {all_video_ids[idx]}, Similarity: {all_similarities[idx]}")
                video_list_22.append(all_video_ids[idx])
            elif all_similarities[idx] >= 21:
                print(f"Video Path: {all_video_ids[idx]}, Similarity: {all_similarities[idx]}")
                video_list_21.append(all_video_ids[idx])
            elif all_similarities[idx] >= 20:
                print(f"Video Path: {all_video_ids[idx]}, Similarity: {all_similarities[idx]}")
                video_list_20.append(all_video_ids[idx])
            elif all_similarities[idx] >= 19:
                print(f"Video Path: {all_video_ids[idx]}, Similarity: {all_similarities[idx]}")
                video_list_19.append(all_video_ids[idx])
            else:
                break
            
        with open('List_25.txt','w') as f:
            for idx in video_list_25:
                f.write(idx + '\n')

        with open('List_24.txt','w') as f:
            for idx in video_list_24:
                f.write(idx + '\n')
        with open('List_23.txt','w') as f:
            for idx in video_list_23:
                f.write(idx + '\n')
        with open('List_22.txt','w') as f:
            for idx in video_list_22:
                f.write(idx + '\n')
        with open('List_21.txt','w') as f:
            for idx in video_list_21:
                f.write(idx + '\n')
        with open('List_20.txt','w') as f:
            for idx in video_list_20:
                f.write(idx + '\n')
        with open('List_19.txt','w') as f:
            for idx in video_list_19:
                f.write(idx + '\n')
            

            
def save_video_embeddings(video_paths, batch_size, num_workers):
    local_rank = parse()  # Get local_rank from environment
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    model_base = CLIP4Clip(model_name='clip_vit_b32',device='cuda:{}'.format(local_rank)) 
    model = DDP(model_base.to(torch.device('cuda:{}'.format(local_rank))), device_ids=[local_rank],output_device=local_rank)

    dataset = VideoDataset(video_paths)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler=sampler)
    
    
    video_emb = []
    video_ids = []

    for batch_frames in dataloader:
        video_paths, frames = batch_frames
        f = model(frames, modality='video')
        video_ids.extend(video_paths)
        video_emb.extend(f)
        
    gathered_emb = [None for _ in range(dist.get_world_size())]
    gathered_video_ids = [None for _ in range(dist.get_world_size())]

    dist.all_gather_object(gathered_emb, video_emb)
    dist.all_gather_object(gathered_video_ids, video_ids)
    all_emb = [sim for sublist in gathered_emb for sim in sublist]
    all_video_ids = [vid for sublist in gathered_video_ids for vid in sublist]
    if int(os.environ['LOCAL_RANK']) == 0:
        np.save('video_embeddings.npy', np.array(all_emb, dtype=np.float32))
        np.save('video_ids.npy', np.array(all_video_ids))    

def main(text_emb):
    local_rank = parse()  # Get local_rank from environment
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    # 视频路径
    video_paths = ['./demoVideo/demo_video.mp4','./demoVide/demo1.mp4','./demoVide/demo2.mp4','./demoVide/woman_dancing.mp4']
    test_path = '/workspace/Crilias/zhangzhenxing/test'
    dataPath = '/workspace/Crilias/zhangzhenxing/dataProcess/data'
    data = '/workspace/Crilias/zhangzhenxing/Embeddings/data'
    EmbeddingsTest = '/workspace/Crilias/zhangzhenxing/Embeddings/EmbeddingsTest'
    video_list = [os.path.join(test_path, video) for video in os.listdir(test_path) if video.endswith('mp4')]
                #  [os.path.join(dataPath, video) for video in os.listdir(dataPath) if video.endswith('mp4')] + \
                #  [os.path.join(data, video) for video in os.listdir(data) if video.endswith('mp4')]
    
    Video_2k = [os.path.join(EmbeddingsTest, video) for video in os.listdir(EmbeddingsTest) if video.endswith('mp4')]

    test = ['/workspace/Crilias/zhangzhenxing/Embeddings/EmbeddingsTest/fc5467b226756c3f54d4b0ab7dc5367d.mp4,']
    originData(local_rank, Video_2k, text_emb, batch_size=16, num_workers=16)
    
    dist.destroy_process_group()

def getText_Video_sim(text):
    model = CLIP4Clip(model_name='clip_vit_b32',device='cuda')
    # text = 'a woman is singing and dancing' 
    text_emb = model(text, modality='text')

    logit_scale = model.model.clip.logit_scale.exp().detach().cpu().numpy()

    video_embeddings = np.load('video_embeddings.npy') 
    video_ids = np.load('video_ids.npy')

    sim = logit_scale * np.matmul(video_embeddings, text_emb)
    top_sim = np.argsort(sim)[::-1]
    for idx in top_sim:
        if sim[idx] >=25:
            print(f"Video Path: {video_ids[idx]}, Similarity: {sim[idx]}")

if __name__ == "__main__":
    video_paths = ['./demoVideo/demo_video.mp4','./demoVide/demo1.mp4','./demoVide/demo2.mp4','./demoVide/woman_dancing.mp4']
    test_path = '/workspace/Crilias/zhangzhenxing/test'
    dataPath = '/workspace/Crilias/zhangzhenxing/dataProcess/data'
    data = '/workspace/Crilias/zhangzhenxing/Embeddings/data'
    EmbeddingsTest = '/workspace/Crilias/zhangzhenxing/Embeddings/EmbeddingsTest'
    Video_2k = [os.path.join(EmbeddingsTest, video) for video in os.listdir(EmbeddingsTest) if video.endswith('mp4')]
    # save_video_embeddings(Video_2k, batch_size=16, num_workers=16)


    text = 'a dog is waiting a people'
    model = CLIP4Clip(model_name='clip_vit_b32',device='cuda')
    text_emb = model(text, modality='text')
    # text = 'a woman is singing and dancing' 
    main(text_emb)