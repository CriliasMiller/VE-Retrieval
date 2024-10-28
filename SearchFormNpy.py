import numpy as np
from EmbeddingsCal import CLIP4Clip

def getText_Video_sim(model, text):
    text_emb = model(text, modality='text')

    logit_scale = model.model.clip.logit_scale.exp().detach().cpu().numpy()

    video_embeddings = np.load('video_embeddings.npy') 
    video_ids = np.load('video_ids.npy')

    sim = logit_scale * np.matmul(video_embeddings, text_emb)
    top_sim = np.argsort(sim)[::-1]
    
    result_text = ""
    for idx in top_sim:
        if sim[idx] >= 25:
            result_text += f"Video Path: {video_ids[idx]}, Similarity: {sim[idx]}\n"
    return result_text

if __name__ == "__main__":
    model = CLIP4Clip(model_name='clip_vit_b32', device='cuda')
    while True:
        text = input("输入查询语句 (输入 'exit' 退出): ")
        if text.lower() == 'exit':
            print("程序已退出。")
            break

        results = getText_Video_sim(model, text)
        print("\nMost similar videos:")
        print(results)