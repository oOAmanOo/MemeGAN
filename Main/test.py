import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from local_gemma import LocalGemma2ForCausalLM

from extractor import addImagePath, textExtraction, imageExtraction, textExtractReverse
eps = torch.finfo(torch.bfloat16).eps
batch_size = 50
### 官方的Gemma #########################################################################################
# 2b = 2304, 9b = 3584, 27b = 4608
gemma_hiddenstate_size = 2304
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
gemmaConfig = AutoConfig.from_pretrained('google/gemma-2-2b-it')
### gemma float32 / bfloat16
gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto", torch_dtype=torch.bfloat16)
### Local gemma
# gemma = LocalGemma2ForCausalLM.from_pretrained("google/gemma-2-2b-it", preset="auto", torch_dtype=torch.bfloat16)
### gemma int4 / int8
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# gemma = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-2-27b-it",
#     quantization_config=quantization_config,
# )
# gemma = LocalGemma2ForCausalLM.from_pretrained(
#     "google/gemma-2-2b-it",
#     quantization_config=quantization_config,
# )
########################################################################################################

class self_multi(nn.Module):
    def __init__(self):
        super(self_multi, self).__init__()
        # self attention
        self.selfAttentionMultihead = nn.MultiheadAttention(768, 1)
        self.selfAttentionLayerNorm = nn.LayerNorm(768, eps=eps)
        self.selfAttentionLinear = nn.Linear(768, 768)
        self.selfAttentionLayerNorm2 = nn.LayerNorm(768, eps=eps)

        # multihead attention
        self.multiheadAttentionMultihead = nn.MultiheadAttention(768, 8)
        self.multiheadAttentionLinear = nn.Linear(768, 768)
        self.multiheadAttentionLayerNorm = nn.LayerNorm(768, eps=eps)

    def forward(self, image, text):
        # self attention module
        self_out = self.selfAttentionMultihead(image, image, image)[0]
        self_out = self.selfAttentionLinear(self_out)
        self_out = self.selfAttentionLayerNorm(self_out + image)

        # multihead attention module
        multi_out = self.multiheadAttentionMultihead(text, text, text)[0]
        multi_out = self.multiheadAttentionLinear(multi_out)
        multi_out = self.multiheadAttentionLayerNorm(multi_out + text)

        return self_out, multi_out


class co_attention(nn.Module):
    def __init__(self):
        super(co_attention, self).__init__()
        # co-attention text
        self.coAttentionTextMultihead = nn.MultiheadAttention(768, 1)
        self.coAttentionTextLinear = nn.Linear(768, 768)
        self.coAttentionTextLayerNorm = nn.LayerNorm(768, eps=eps)

        # co-attention image
        self.coAttentionImageMultihead = nn.MultiheadAttention(768, 1)
        self.coAttentionImageLinear = nn.Linear(768, 768)
        self.coAttentionImageLayerNorm = nn.LayerNorm(768, eps=eps)

    def forward(self, image, text):
        # co-attention image module
        visual_attending_textual = self.coAttentionTextMultihead(image, text, text)[0]
        visual_attending_textual = self.coAttentionTextLinear(visual_attending_textual)
        visual_attending_textual = self.coAttentionTextLayerNorm(visual_attending_textual + image)

        # co-attention text module
        textual_attending_visual = self.coAttentionTextMultihead(text, image, image)[0]
        textual_attending_visual = self.coAttentionTextLinear(textual_attending_visual)
        textual_attending_visual = self.coAttentionTextLayerNorm(textual_attending_visual + text)

        return visual_attending_textual, textual_attending_visual


class Generator(nn.Module):
    def __init__(self, depth=12):
        super(Generator, self).__init__()
        self.layers_self_multi = nn.ModuleList([self_multi() for _ in range(depth)])
        self.layers_co_attention = nn.ModuleList([co_attention() for _ in range(depth)])

        # feed forward
        self.feedForwardLinear = nn.Linear(768, 768)
        self.feedForwardLayerNorm = nn.LayerNorm(768, eps=eps)

        # gemma
        self.gemmaLinearMaxTokens = nn.Linear(64, 16)
        self.gemmaLinearBefore = nn.Linear(768, gemmaConfig.vocab_size)
        self.gemmaSoftmax = nn.Softmax(dim=2)
        self.gemma = nn.Sequential(*list(gemma.children())[:-1])
        # self.gemmaLm_head = nn.Sequential(*list(gemma.children())[1:])
        self.gemmaLm_headbf = nn.Linear(768, gemma_hiddenstate_size)
        self.gemmaLm_head = nn.Linear(gemma_hiddenstate_size, gemmaConfig.vocab_size)

        # funny score
        self.FunnyScorelinear1 = nn.Linear(768, 1)
        self.FunnyScorelinear2 = nn.Linear(64, 1)

    def gemmaGenerate(self, x):
        with torch.no_grad():
            # maximum 32 tokens
            x = self.gemmaLinearMaxTokens(x.transpose(1, 2)).transpose(1, 2)
            x = self.gemmaLinearBefore(x)
            x2 = self.gemmaSoftmax(x + eps)

            # get max value of each row, total 32*64
            top_k_values, top_k_indices = torch.topk(x2, 1, dim=2, largest=True)
            toGemma = textExtractReverse(gemma, tokenizer, top_k_indices).to(device)
            # 使用gemma作為model的一部分
            output = self.gemma(toGemma)
            # output[0] = last_hidden_state
            # output[1] = past_key_values

        return output[0]

    def forward(self, text, image):
        # max_seq_len = max(text.shape[1], image.shape[1])
        # text = nn.functional.pad(text, (0, 0, 0, max_seq_len - text.shape[1]))
        # image = nn.functional.pad(image, (0, 0, 0, max_seq_len - image.shape[1]))
        text = text.transpose(0, 1)
        image = image.transpose(0, 1)

        ######################### Transformer #########################
        for self_multi_layer in self.layers_self_multi:
            image, text = self_multi_layer(image, text)
        for co_attention_layer in self.layers_co_attention:
            image, text = co_attention_layer(image, text)
        ###############################################################

        # feature fusion
        feature_fusion = image + text  # visual_attending_textual + textual_attending_visual
        feature_fusionFF = self.feedForwardLinear(feature_fusion)
        feature_fusion_final = self.feedForwardLayerNorm(feature_fusion + feature_fusionFF)
        feature_fusion_final = feature_fusion_final.squeeze(-1)
        feature_fusion_final = feature_fusion_final.transpose(0, 1)
        ####################### gemma  generate #######################
        # last_hidden_state = self.gemmaGenerate(feature_fusion_final)
        # output_text = self.gemmaLm_head(last_hidden_state)
        ###############################################################
        output_text = self.gemmaLm_headbf(feature_fusion_final)
        output_text = self.gemmaLm_head(output_text)
        ###############################################################
        output_text = output_text.to(torch.bfloat16)
        ######################### funny score #########################
        output_funny_score = self.FunnyScorelinear1(feature_fusion_final).squeeze(-1)
        output_funny_score = self.FunnyScorelinear2(output_funny_score).squeeze(-1)
        ###############################################################

        return output_text, output_funny_score

    def generate(self, image, max_length=100):
        generated_tokens = []
        generated_tokens.append(2)  # <bos> = 2
        text = torch.zeros_like(image).to(device)
        text = text.transpose(0, 1)
        image = image.transpose(0, 1)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        depth = len(self.layers_self_multi)

        # 有時後空格會失效，所以手動插入空格 <pad> = 0
        def insert_zeros(list):
            zeros = [0] * (2 * len(list) - 1)
            zeros[::2] = list
            return zeros

        lastTurn = False
        with torch.no_grad():
            for _ in range(max_length + 1):
                # Transformer
                for i in range(depth):
                    # self attention
                    image, text = self.layers_self_multi[i](image, text)
                    # co-attention
                    image, text = self.layers_co_attention[i](image, text)

                # feature fusion
                feature_fusion = image + text  # visual_attending_textual + textual_attending_visual
                feature_fusionFF = self.feedForwardLinear(feature_fusion)
                feature_fusion_final = self.feedForwardLayerNorm(feature_fusion + feature_fusionFF)
                feature_fusion_final = feature_fusion_final.squeeze(-1)
                feature_fusion_final = feature_fusion_final.transpose(0, 1)
                ####################### gemma  generate #######################
                # last_hidden_state = self.gemmaGenerate(feature_fusion_final)
                # output_text = self.gemmaLm_head(last_hidden_state)
                ###############################################################
                output_text = self.gemmaLm_headbf(feature_fusion_final)
                output_text = self.gemmaLm_head(output_text)
                ###############################################################
                output_text = output_text.to(torch.bfloat16)

                # funny score
                output_funny_score = self.FunnyScorelinear1(feature_fusion_final).squeeze(-1)
                output_funny_score = self.FunnyScorelinear2(output_funny_score).squeeze(-1)

                if lastTurn:  # show final funny score
                    return generated_caption, output_funny_score
                else:
                    next_token_logits = output_text[:, -1, :]
                    next_token_probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.argmax(next_token_probs, dim=-1).item()
                    generated_tokens.append(next_token_id)

                    generated_caption = insert_zeros(generated_tokens)
                    generated_caption = tokenizer.decode(generated_caption, skip_special_tokens=False)
                    generated_caption = generated_caption.replace("<pad>", " ").replace("  ", " ").split()
                    generated_caption = [word for word in generated_caption if word[0] != "<"]
                    generated_caption = " ".join(generated_caption)
                    print(generated_caption)

                    text = textExtraction(tokenizer, gemmaConfig, [generated_caption]).to(device).to(torch.bfloat16)
                    text = text.transpose(0, 1)

                    if next_token_id in gemmaConfig.eos_token_id or len(generated_caption.split()) > max_length:
                        # <eos> = 1; <end_of_turn> = 107
                        lastTurn = True
        return generated_caption, output_funny_score


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linearFake = nn.Linear(gemmaConfig.vocab_size, 768)
        # Generator
        self.g_con_mlp1 = nn.Linear(1536, 2)
        self.g_con_mlp2 = nn.Linear(64, 1)
        self.g_unc_mlp1 = nn.Linear(768, 1)
        self.g_unc_mlp2 = nn.Linear(64, 1)
        # Discriminator
        self.d_linearFake = nn.Linear(gemmaConfig.vocab_size, 768)
        self.d_con_mlp1_r2f = nn.Linear(3072, 2)
        self.d_con_mlp2_r2f = nn.Linear(64, 1)
        self.d_con_mlp1_f2r = nn.Linear(3072, 2)
        self.d_con_mlp2_f2r = nn.Linear(64, 1)
        self.d_con_mlp1_g = nn.Linear(1536, 2)
        self.d_con_mlp2_g = nn.Linear(64, 1)
        self.d_con_mlp1_m = nn.Linear(1536, 2)
        self.d_con_mlp2_m = nn.Linear(64, 1)
        self.d_unc_mlp1_r = nn.Linear(768, 1)
        self.d_unc_mlp2_r = nn.Linear(64, 1)
        self.d_unc_mlp1_g = nn.Linear(768, 1)
        self.d_unc_mlp2_g = nn.Linear(64, 1)
        self.d_unc_mlp1_m = nn.Linear(768, 1)
        self.d_unc_mlp2_m = nn.Linear(64, 1)

    def forward(self, real_text, fake_text, image, GorD):
        # real_text = [batch_size, 64, 768]
        # fake_text = [batch_size, 64, 256000]
        # image = [batch_size, 64, 768]
        fake_text = self.linearFake(fake_text)
        if GorD == "G":
            g_C_g = torch.cat((fake_text, image), dim=-1)
            ########################  conditional  ########################
            g_C_g = self.g_con_mlp1(g_C_g)
            g_C_g = self.g_con_mlp2(g_C_g.transpose(1, 2)).squeeze(-1)
            ###############################################################
            ######################## unconditional ########################
            g_UC_g = self.g_unc_mlp1(fake_text).squeeze(-1)
            g_UC_g = self.g_unc_mlp2(g_UC_g).squeeze(-1)
            ###############################################################
            return g_C_g, g_UC_g

        elif GorD == "D":
            mismatched_text = torch.roll(real_text, 1, 0)
            C_r = torch.cat((real_text, image), dim=-1)
            C_g = torch.cat((fake_text, image), dim=-1)
            C_m = torch.cat((mismatched_text, image), dim=-1)
            # contrastive discriminator
            cd_C_r = C_r.unsqueeze(0).expand(C_r.shape[0], -1, -1, -1)
            cd_C_g = C_g.unsqueeze(0).expand(C_g.shape[0], -1, -1, -1)
            d_C_r2f = torch.cat((cd_C_r, cd_C_g.transpose(0, 1)), dim=-1)
            d_C_f2r = torch.cat((cd_C_g, cd_C_r.transpose(0, 1)), dim=-1)

            ######################## conditional ########################
            d_C_r2f = self.d_con_mlp1_r2f(d_C_r2f)
            d_C_f2r = self.d_con_mlp1_f2r(d_C_f2r)
            d_C_g = self.d_con_mlp1_g(C_g)
            d_C_m = self.d_con_mlp1_m(C_m)

            d_C_r2f = self.d_con_mlp2_r2f(d_C_r2f.transpose(2, 3)).squeeze(-1).unsqueeze(0)
            d_C_f2r = self.d_con_mlp2_r2f(d_C_f2r.transpose(2, 3)).squeeze(-1).unsqueeze(0)
            d_C_g = self.d_con_mlp2_g(d_C_g.transpose(1, 2)).squeeze(-1).unsqueeze(0)
            d_C_m = self.d_con_mlp2_m(d_C_m.transpose(1, 2)).squeeze(-1).unsqueeze(0)

            d_C_r2f = torch.mean(d_C_r2f, dim=-2)
            d_C_f2r = torch.mean(d_C_f2r, dim=-2)

            d_con_output = torch.cat((d_C_r2f, d_C_f2r, d_C_g, d_C_m), dim=0)
            ###############################################################

            ######################## unconditional ########################
            d_UC_r = self.d_unc_mlp1_r(real_text).squeeze(-1)
            d_UC_g = self.d_unc_mlp1_g(fake_text).squeeze(-1)
            d_UC_m = self.d_unc_mlp1_m(mismatched_text).squeeze(-1)

            d_UC_r = self.d_unc_mlp2_r(d_UC_r).squeeze(-1).unsqueeze(0)
            d_UC_g = self.d_unc_mlp2_g(d_UC_g).squeeze(-1).unsqueeze(0)
            d_UC_m = self.d_unc_mlp2_m(d_UC_m).squeeze(-1).unsqueeze(0)

            d_unc_output = torch.cat((d_UC_r, d_UC_g, d_UC_m), dim=0)
            ###############################################################
            return d_con_output, d_unc_output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#load model
NetG = Generator().to(torch.bfloat16).to(device)
NetD = Discriminator().to(torch.bfloat16).to(device)
checkpoint_G = torch.load('./Model/20241029_nogemma/20241029_nogemma_NetG_12.pth')
checkpoint_D = torch.load('./Model/20241029_nogemma/20241029_nogemma_NetD_12.pth')
NetG.load_state_dict(checkpoint_G['model_state_dict'])
NetD.load_state_dict(checkpoint_D['model_state_dict'])
# optimizer_G.load_state_dict(checkpoint_G['optimizer_state_dict'])
# optimizer_D.load_state_dict(checkpoint_D['optimizer_state_dict'])
# train_losses_FC.append(checkpoint_G['FC_loss'])
# train_losses_G.append(checkpoint_G['G_loss'])
# train_losses_D.append(checkpoint_G['D_loss'])
# present_epoch = checkpoint_G['epoch'] + 1
# train with load model
NetG.train()
NetD.train()
# generate
NetG.eval()
NetD.eval()
image = imageExtraction("./test_img.jpg")
output = NetG.generate(image.to(device).to(torch.bfloat16), 200)
print(output)