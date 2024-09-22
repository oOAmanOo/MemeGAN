import torch.nn as nn
import pandas as pd
from torch.distributed.pipelining import pipeline

from transformers import pipeline
from transformers import AutoTokenizer
from tqdm import tqdm
from transformers import AutoImageProcessor, Swinv2Model
from PIL import Image
import torch

def addImagePath(data, imgPath):
    data['image_id'] = data['image_id'].apply(lambda x: imgPath + str(x) + '.jpg')
    return data

def textExtraction(text_data):
    # 載入模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emoji", device=0)
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emoji")
    vocab_size = 50265  # 词汇表大小
    embedding_dim = 768  # 嵌入维度，与你的图像嵌入维度相同
    text_embedding = nn.Embedding(vocab_size, embedding_dim).to(device)

    all_features = []
    with tqdm(text_data) as pbar:
        for text in (text_data):
            tokens = tokenizer(text, truncation=True, padding='max_length', max_length=373, return_tensors='pt')
            output = text_embedding(tokens['input_ids'].to(device))
            output = output.cpu()
            all_features.append(output)
            pbar.update(1)
        return torch.cat(all_features)


# 定義批量處理和提取特徵的函數
def imageExtraction(image_data):
    # 加載 Swinv2 模型和處理器
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    swin = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

    # 將模型設置為評估模式
    swin.eval()

    # 如果有 GPU，可以將模型移動到 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    swin.to(device)

    all_features = []
    with tqdm(image_data) as pbar:
        for image_path in (image_data):
            with torch.no_grad():
                # 加載並預處理圖像
                image = Image.open(image_path)
                # 使用 image_processor 將 batch 圖片處理成適合模型的格式
                inputs = image_processor(image, return_tensors="pt")
                # 將 inputs 放到 GPU 上（如果可用）
                inputs.to(device)
                # 獲取 Swinv2 模型的輸出
                outputs = swin(**inputs)
                last_hidden_states = outputs.last_hidden_state
                # print(last_hidden_states.shape)
                # 儲存特徵
                last_hidden_states = last_hidden_states.cpu()
                all_features.append(last_hidden_states)
                pbar.update(1)
    return torch.cat(all_features)

