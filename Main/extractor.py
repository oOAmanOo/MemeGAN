from xml.sax.handler import all_features

from lib2to3.btm_utils import tokens

import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm
from transformers import AutoImageProcessor, Swinv2Model
import torch.nn.functional as F
from PIL import Image
import torch

def addImagePath(data, imgPath):
    data['image_id'] = data['image_id'].apply(lambda x: imgPath + str(x) + '.jpg')
    return data

def textExtraction(text_data):
    ##### google/gemma-2b #####
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    # vocab_size = 256128  # 词汇表大小

    ###### twitter-roberta-base-emoji ######
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emoji")
    vocab_size = 50265  # 詞彙表大小

    embedding_dim = 768  # 嵌入维度，與你的圖片嵌入维度相同
    text_embedding = nn.Embedding(vocab_size, embedding_dim)

    all_features = []
    with tqdm(text_data) as pbar:
        for text in (text_data):
            tokens = tokenizer(text, padding='longest', return_tensors='pt')
            output = text_embedding(tokens['input_ids'])
            linear = torch.nn.Linear(output.shape[1], 64)
            projected_output = linear(output.transpose(1, 2)).transpose(1, 2)
            all_features.append(projected_output)
            pbar.update(1)
        return torch.cat(all_features)

def textExtractReverse(data):
    ##### google/gemma-2b #####
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

    ###### twitter-roberta-base-emoji ######
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emoji")
    # reverse the token
    reverse = tokenizer.batch_decode(data.squeeze(-1), skip_special_tokens=True)
    # tokenize with gemma-2b
    tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-2b")
    prompt = "Create a funny meme using the following text as the foundation for the joke. The meme should creatively incorporate humor that is relatable and witty, matching the tone of the provided content. Make sure to blend the text with an amusing visual representation that enhances the punchline: "
    all_features = []
    for i, text in enumerate(reverse):
        reverse[i] = prompt + text
    tokens = tokenizer_gemma(reverse, padding='max_length', max_length=256, return_tensors='pt')
    all_features.append(tokens['input_ids'])
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
                # 儲存特徵
                last_hidden_states = last_hidden_states.cpu()
                all_features.append(last_hidden_states)
                pbar.update(1)
    return torch.cat(all_features)

