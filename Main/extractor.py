import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.distributed.pipelining import pipeline
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer
from transformers import BertModel
from sklearn.model_selection import train_test_split
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from transformers import AutoImageProcessor, Swinv2Model
from PIL import Image
import torch
from torch.utils.data import DataLoader

def text_extraction(data, imgPath):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emoji", device=0)
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emoji")

    # 将标题字段转换为tokens
    def tokenize_bert(words):
        # 使用BERT Tokenizer对标题进行tokenize
        tokens = tokenizer(words, truncation=True, padding='max_length', max_length=768,return_tensors='pt')
        return tokens['input_ids'].tolist()[0]

    new_data = pd.DataFrame()
    new_data['text'] = data['caption'].apply(tokenize_bert)
    new_data = new_data['text'].apply(pd.Series)
    new_data.columns = ['text_{}'.format(i) for i in range(new_data.shape[1])]
    new_data['image_id'] = data['image_id'].apply(lambda x: imgPath + str(x) + '.jpg')
    new_data['funny_score'] = data['funny_score']

    return new_data


# 定義批量處理和提取特徵的函數
def image_extraction(image_data):
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

