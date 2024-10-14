import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoImageProcessor, Swinv2Model
from PIL import Image
import torch
from transformers import AutoConfig

def addImagePath(data, imgPath):
    data['image_id'] = data['image_id'].apply(lambda x: imgPath + str(x) + '.jpg')
    return data

def textExtraction(text_data):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    gemmaConfig = AutoConfig.from_pretrained('google/gemma-2-2b-it')
    vocab_size = gemmaConfig.vocab_size  # 詞彙表大小

    embedding_dim = 768  # 嵌入维度，與你的圖片嵌入维度相同
    text_embedding = nn.Embedding(vocab_size, embedding_dim)

    all_features = []
    for text in (text_data):
        tokens = tokenizer(text, padding='longest', return_tensors='pt')
        output = text_embedding(tokens['input_ids'])
        linear = torch.nn.Linear(output.shape[1], 64)
        projected_output = linear(output.transpose(1, 2)).transpose(1, 2)
        all_features.append(projected_output)
    return torch.cat(all_features)

def textExtractReverse(data):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    # 有時後空格會失效，所以手動插入空格 <pad> = 0
    def insert_zeros(tensor):
        zeros = torch.zeros(tensor.shape[0], tensor.shape[1] * 2 - 1)
        zeros[:, ::2] = tensor
        zeros = zeros.to(int)
        return zeros

    reverse_data = insert_zeros(data.squeeze(-1))
    # reverse the token
    reverse = tokenizer.batch_decode(reverse_data, skip_special_tokens=False)
    # tokenize with gemma-2b
    tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    prompt = "Write a humor memetic post for Instagram with the following elements: "
    # all_features = []
    for i, text in enumerate(reverse):
        text = text.replace("<pad>", " ").replace("  ", " ")
        text = text.split()
        text = ', '.join(text)
        reverse[i] = prompt + text + "."
    tokens = tokenizer_gemma(reverse, padding='max_length', max_length=64, return_tensors='pt')
    # all_features.append(tokens['input_ids'])
    return tokens['input_ids']

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
    for image_path in (image_data):
        with torch.no_grad():
            # 加載並預處理圖像
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            image = image[:, :, :3]
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
    return torch.cat(all_features)

