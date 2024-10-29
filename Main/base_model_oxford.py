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

class OxfordDataset(torch.utils.data.Dataset):
    def __init__(self, text, image, funny_score):
        self.text = text
        self.image = image
        self.funny_score = funny_score

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        imageData = torch.load('../../Oxford_HIC/ImageData/'+ self.image[idx] +'.pt', weights_only=False)
        # all dtype to torch.float16
        imageData = imageData

        return self.text[idx], imageData, self.funny_score[idx]


def train():
    epochs = 30
    batch_size = 50
    optimizer_G_lr = 1e-5
    optimizer_D_lr = 1e-5
    save_name = '20241029_loss_new'
    # save_name = '20241028'
    if not os.path.exists('./Model/' + save_name):
        os.makedirs('./Model/' + save_name)


    # if args.img - dir == 'Oxford_HIC':
    dirPath = '../Data/Oxford_HIC/CaptionID_oxford_hic_data.csv'
    # else:
    # dirPath = '../Data/Instagram/Filter_' + 'wendys' + '.csv'
    # imgPath = '../Data/Instagram/' + 'wendys' + '_img/'
    # load data
    data = pd.read_csv(dirPath)
    print("shape of data: ", data.shape)
    data = data.sample(n=50000, random_state=42, replace=True).reset_index(drop=True)
    # frac = 0.05 ==> 5% of the data = 169904
    # n = 169920 ==> 72 * 2360 = 169920 (F2G)
    # n = 169988 ==> 91 * 1868 = 169988 (G2F)
    print("sample of data: ", data.shape)

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train_text = train['caption'].tolist()
    train_image = train['image_id'].tolist()
    train_funny_score = train['funny_score'].tolist()
    test_text = test['caption'].tolist()
    test_image = test['image_id'].tolist()
    test_funny_score = test['funny_score'].tolist()

    train_dataset = OxfordDataset(train_text, train_image, train_funny_score)
    test_dataset = OxfordDataset(test_text, test_image, test_funny_score)
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=20)
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=20)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True, drop_last=True)

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
            self.gemmaLm_head = nn.Linear(gemma_hiddenstate_size,gemmaConfig.vocab_size)

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
                toGemma = textExtractReverse(top_k_indices).to(device)
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
            last_hidden_state = self.gemmaGenerate(feature_fusion_final)
            output_text = self.gemmaLm_head(last_hidden_state)
            ###############################################################
            # output_text = self.gemmaLm_headbf(feature_fusion_final)
            # output_text = self.gemmaLm_head(output_text)
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
                    feature_fusion = self.feedForwardLinear(feature_fusion)
                    feature_fusion = self.feedForwardLayerNorm(feature_fusion + feature_fusion)
                    feature_fusion = feature_fusion.squeeze(-1)
                    feature_fusion = feature_fusion.transpose(0, 1)

                    # gemma generate
                    last_hidden_state = self.gemmaGenerate(feature_fusion)
                    output_text = self.gemmaLm_head(last_hidden_state)

                    # funny score
                    output_funny_score = self.FunnyScorelinear1(feature_fusion).squeeze(-1)
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

                        text = textExtraction([generated_caption]).to(device)
                        text = text.transpose(0, 1)

                        if next_token_id in gemmaConfig.eos_token_id or len(generated_caption.split()) > max_length:
                            # <eos> = 1; <end_of_turn> = 107
                            lastTurn = True

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
            self.d_con_mlp1_r2f = nn.Linear(196608, 2)
            self.d_con_mlp2_r2f = nn.Linear(batch_size, 1)
            self.d_con_mlp1_f2r = nn.Linear(196608, 2)
            self.d_con_mlp2_f2r = nn.Linear(batch_size, 1)
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
                d_C_r2f = d_C_r2f.view(d_C_r2f.shape[0], d_C_r2f.shape[1], -1)
                d_C_f2r = d_C_f2r.view(d_C_f2r.shape[0], d_C_f2r.shape[1], -1)

                ######################## conditional ########################
                d_C_r2f = self.d_con_mlp1_r2f(d_C_r2f)
                d_C_f2r = self.d_con_mlp1_f2r(d_C_f2r)
                d_C_g = self.d_con_mlp1_g(C_g)
                d_C_m = self.d_con_mlp1_m(C_m)

                d_C_r2f = self.d_con_mlp2_r2f(d_C_r2f.transpose(1,2)).squeeze(-1).unsqueeze(0)
                d_C_f2r = self.d_con_mlp2_r2f(d_C_f2r.transpose(1,2)).squeeze(-1).unsqueeze(0)
                d_C_g = self.d_con_mlp2_g(d_C_g.transpose(1,2)).squeeze(-1).unsqueeze(0)
                d_C_m = self.d_con_mlp2_m(d_C_m.transpose(1,2)).squeeze(-1).unsqueeze(0)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

    NetG = Generator().to(torch.bfloat16).to(device)
    NetD = Discriminator().to(torch.bfloat16).to(device)
    optimizer_G = optim.Adam(NetG.parameters(), lr=optimizer_G_lr)
    optimizer_D = optim.Adam(NetD.parameters(), lr=optimizer_D_lr)

    train_losses_G = []
    train_losses_D = []
    test_losses_G = []
    test_losses_D = []
    save = []
    present_epoch = 1
    best_train_loss_G = 9999
    best_train_loss_D = 9999
    best_test_loss_G = 9999
    best_test_loss_D = 9999

    # # separate loss
    # g_fc_loss_list = []
    # g_con_loss_list = []
    # g_unc_loss_list = []
    # d_con_r2f_loss_list = []
    # d_con_f2r_loss_list = []
    # d_con_f_loss_list = []
    # d_con_m_loss_list = []
    # d_unc_r_loss_list = []
    # d_unc_f_loss_list = []
    # d_unc_m_loss_list = []

    checkpoint = False
    if checkpoint:
        checkpoint_G = torch.load('./Model/test_save/test_save_2NetG.pth')
        checkpoint_D = torch.load('./Model/test_save/test_save_2NetD.pth')
        NetG.load_state_dict(checkpoint_G['model_state_dict'])
        NetD.load_state_dict(checkpoint_D['model_state_dict'])
        optimizer_G.load_state_dict(checkpoint_G['optimizer_state_dict'])
        optimizer_D.load_state_dict(checkpoint_D['optimizer_state_dict'])
        train_losses_G.append(checkpoint_G['G_loss'])
        train_losses_D.append(checkpoint_G['D_loss'])
        present_epoch = checkpoint_G['epoch'] + 1

    def funnyScoreLoss(output_funny_score, funny_score):
        loss = nn.MSELoss()(output_funny_score, funny_score)
        # g_fc_loss_list.append(loss.item())
        return loss

    def generatorLoss(condition_logits, uncondition_logits):
        result_fake_con = torch.zeros(condition_logits.shape[0]).to(torch.long).to(device)
        result_fake_unc = torch.FloatTensor(condition_logits.shape[0]).to(torch.bfloat16).uniform_(0.0, 0.1).to(device)
        con_loss = CrossEntropyLoss(label_smoothing=0.1)(condition_logits, result_fake_con)
        unc_loss = BCEWithLogitsLoss()(uncondition_logits, result_fake_unc)
        # g_con_loss_list.append(con_loss.item())
        # g_unc_loss_list.append(unc_loss.item())
        loss = con_loss + unc_loss
        return loss

    def discriminatorLoss(condition_logits, uncondition_logits):
        result_true = torch.FloatTensor(uncondition_logits[0].shape[0]).to(torch.bfloat16).uniform_(0.9, 1.0).to(device)
        result_fake_ce = torch.zeros(uncondition_logits[0].shape[0]).to(torch.long).to(device)
        result_fake_unc_f = torch.FloatTensor(uncondition_logits[1].shape[0]).to(torch.bfloat16).uniform_(0.0, 0.1).to(device)
        result_fake_unc_m = torch.FloatTensor(uncondition_logits[2].shape[0]).to(torch.bfloat16).uniform_(0.0, 0.1).to(device)

        con_r2f = CrossEntropyLoss(label_smoothing=0.1)(condition_logits[0], result_fake_ce)
        con_f2r = CrossEntropyLoss(label_smoothing=0.1)(condition_logits[1], result_fake_ce)
        con_f = CrossEntropyLoss(label_smoothing=0.1)(condition_logits[2], result_fake_ce)
        con_m = CrossEntropyLoss(label_smoothing=0.1)(condition_logits[3], result_fake_ce)
        unc_r = BCEWithLogitsLoss()(uncondition_logits[0], result_true)
        unc_f = BCEWithLogitsLoss()(uncondition_logits[1], result_fake_unc_f)
        unc_m = BCEWithLogitsLoss()(uncondition_logits[2], result_fake_unc_m)
        # d_con_r2f_loss_list.append(con_r2f.item())
        # d_con_f2r_loss_list.append(con_f2r.item())
        # d_con_f_loss_list.append(con_f.item())
        # d_con_m_loss_list.append(con_m.item())
        # d_unc_r_loss_list.append(unc_r.item())
        # d_unc_f_loss_list.append(unc_f.item())
        # d_unc_m_loss_list.append(unc_m.item())
        loss = ((con_r2f + con_f2r) / 2) + ((con_f + con_m) / 2) + unc_r + ((unc_f + unc_m) / 2)
        return loss



    startTime = 0
    textExtractionTime = 0
    GeneratorForwardTime = 0
    GeneratorBackwardGTime = 0
    DiscriminatorForwardTime = 0
    DiscriminatorBackwardTime = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        print("---------------------------------------- epoch " + str(
            epoch + present_epoch) + " ---------------------------------------")
        train_loss_G = 0
        train_loss_D = 0
        test_loss_G = 0
        test_loss_D = 0
        pre = 0
        ###################################### Train ######################################
        with tqdm(train_loader, unit="batch", leave=True) as tepoch:
            for idx, (text, image, funny_score) in enumerate(tepoch):
                # tepoch.set_postfix({'Now': tepoch.format_dict['elapsed'], 'Status': " New batch preprocessing"})
                if idx == 0:
                    startTime = tepoch.format_dict['elapsed']
                elif idx == 1:
                    startTime = (tepoch.format_dict['elapsed'] - pre) * 2
                else:
                    startTime += tepoch.format_dict['elapsed'] - pre
                pre = tepoch.format_dict['elapsed']
                text = textExtraction(tokenizer, gemmaConfig, text).to(torch.bfloat16)
                image = image.to(torch.bfloat16)
                funny_score = funny_score.to(torch.bfloat16)
                # print(text.dtype, image.dtype, funny_score.dtype)
                # print(text.shape, image.shape, funny_score.shape)
                # torch.Size([32, 64, 768]) torch.Size([32, 64, 768]) torch.Size([32, 1])
                textExtractionTime += tepoch.format_dict['elapsed'] - pre
                pre = tepoch.format_dict['elapsed']
                ######################################################
                # (1) Update Generator network
                ######################################################
                optimizer_G.zero_grad()
                # tepoch.set_postfix({'Now': tepoch.format_dict['elapsed'], 'Status': " Generator Forward"})
                logits, output_funny_score = NetG(text.to(device), image.to(device))
                g_con_logits, g_unc_logits = NetD(text.to(device), logits, image.to(device), "G")
                GeneratorForwardTime += tepoch.format_dict['elapsed'] - pre
                pre = tepoch.format_dict['elapsed']

                # tepoch.set_postfix({'Now': tepoch.format_dict['elapsed'], 'Status': " Generator Backward - Generator"})
                loss_G = generatorLoss(g_con_logits.to(device), g_unc_logits.to(device)) + (180 * funnyScoreLoss(output_funny_score.to(device), funny_score.to(device)))
                loss_G.backward(retain_graph=True)
                train_loss_G += loss_G.item()
                optimizer_G.step()
                GeneratorBackwardGTime += tepoch.format_dict['elapsed'] - pre
                pre = tepoch.format_dict['elapsed']
                ######################################################
                # (4) Update Discriminator network
                ######################################################
                optimizer_D.zero_grad()
                # tepoch.set_postfix({'Now': tepoch.format_dict['elapsed'], 'Status': " Discriminator Forward"})
                d_con_logits, d_unc_logits = NetD(text.to(device).detach(), logits.detach(), image.to(device).detach(),
                                                  "D")
                # tepoch.set_postfix({'Now': tepoch.format_dict['elapsed'], 'Status': " Discriminator Backward"})
                DiscriminatorForwardTime += tepoch.format_dict['elapsed'] - pre
                pre = tepoch.format_dict['elapsed']
                loss_D = discriminatorLoss(d_con_logits.to(device), d_unc_logits.to(device))
                loss_D.backward()
                DiscriminatorBackwardTime += tepoch.format_dict['elapsed'] - pre
                pre = tepoch.format_dict['elapsed']
                optimizer_D.step()
                train_loss_D += loss_D.item()
                ######################################################
                # tepoch.set_postfix({'FC_loss': train_loss_FC/ (idx+1), 'G_loss': train_loss_G/ (idx+1), 'D_loss': train_loss_D/ (idx+1)})
                tepoch.set_postfix({'start': startTime / (idx + 1), 'textExtraction': textExtractionTime / (idx + 1),
                                    'GeneratorForward': GeneratorForwardTime / (idx + 1),
                                    'GeneratorBackwardG': GeneratorBackwardGTime / (idx + 1),
                                    'DiscriminatorForward': DiscriminatorForwardTime / (idx + 1),
                                    'DiscriminatorBackward': DiscriminatorBackwardTime / (idx + 1)})
                # sepateLoss = pd.DataFrame()
                # sepateLoss['g_fc_loss'] = g_fc_loss_list
                # sepateLoss['g_con_loss'] = g_con_loss_list
                # sepateLoss['g_unc_loss'] = g_unc_loss_list
                # sepateLoss['d_con_r2f_loss'] = d_con_r2f_loss_list
                # sepateLoss['d_con_f2r_loss'] = d_con_f2r_loss_list
                # sepateLoss['d_con_f_loss'] = d_con_f_loss_list
                # sepateLoss['d_con_m_loss'] = d_con_m_loss_list
                # sepateLoss['d_unc_r_loss'] = d_unc_r_loss_list
                # sepateLoss['d_unc_f_loss'] = d_unc_f_loss_list
                # sepateLoss['d_unc_m_loss'] = d_unc_m_loss_list
                # sepateLoss.to_csv('./Model/' + save_name + "/" + save_name + '_sepateLoss.csv', index=False)
                ######################################################
        train_loss_G /= len(train_loader)
        train_loss_D /= len(train_loader)
        train_losses_G.append(train_loss_G)
        train_losses_D.append(train_loss_D)
        ###################################### Train ######################################

        ######################################  Test ######################################
        with tqdm(test_loader, unit="batch", leave=True) as tepoch:
            for idx, (text, image, funny_score) in enumerate(tepoch):
                text = textExtraction(tokenizer, gemmaConfig, text).to(torch.bfloat16)
                image = image.to(torch.bfloat16)
                funny_score = funny_score.to(torch.bfloat16)
                # Generator
                logits, output_funny_score = NetG(text.to(device), image.to(device))
                # Discriminator
                g_con_logits, g_unc_logits = NetD(text.to(device), logits, image.to(device), "G")
                d_con_logits, d_unc_logits = NetD(text.to(device).detach(), logits.detach(), image.to(device).detach(),"D")
                # loss
                loss_G = generatorLoss(g_con_logits.to(device), g_unc_logits.to(device)) + (
                            180 * funnyScoreLoss(output_funny_score.to(device), funny_score.to(device)))
                loss_D = discriminatorLoss(d_con_logits, d_unc_logits)
                test_loss_G += loss_G.item()
                test_loss_D += loss_D.item()
                tepoch.set_postfix({'G_loss': test_loss_G / (idx + 1),
                                    'D_loss': test_loss_D / (idx + 1)})
        test_loss_G /= len(test_loader)
        test_loss_D /= len(test_loader)
        test_losses_G.append(test_loss_G)
        test_losses_D.append(test_loss_D)
        ######################################  Test ######################################

        ######################################  Save ######################################
        hasSaved = False
        # 任一個loss小於最佳loss就存檔
        if train_loss_G < best_train_loss_G and test_loss_G < best_test_loss_G:
            best_train_loss_G = train_loss_G
            best_test_loss_G = test_loss_G
            hasSaved = True
            torch.save({
                'epoch': epoch + present_epoch,
                'model_state_dict': NetG.state_dict(),
                'optimizer_state_dict': optimizer_G.state_dict(),
                'G_loss': loss_G,
                'D_loss': loss_D,
            }, './Model/' + save_name + "/" + save_name + '_NetG_' + str(epoch + present_epoch) + '.pth')
            torch.save({
                'epoch': epoch + present_epoch,
                'model_state_dict': NetD.state_dict(),
                'optimizer_state_dict': optimizer_D.state_dict(),
                'G_loss': loss_G,
                'D_loss': loss_D,
            }, './Model/' + save_name + "/" + save_name + '_NetD_' + str(epoch + present_epoch) + '.pth')
        if train_loss_D < best_train_loss_D and test_loss_D < best_test_loss_D:
            best_train_loss_D = train_loss_D
            best_test_loss_D = test_loss_D
            hasSaved = True
            torch.save({
                'epoch': epoch + present_epoch,
                'model_state_dict': NetG.state_dict(),
                'optimizer_state_dict': optimizer_G.state_dict(),
                'G_loss': loss_G,
                'D_loss': loss_D,
            }, './Model/' + save_name + "/" + save_name + '_NetG_' + str(epoch + present_epoch) + '.pth')
            torch.save({
                'epoch': epoch + present_epoch,
                'model_state_dict': NetD.state_dict(),
                'optimizer_state_dict': optimizer_D.state_dict(),
                'G_loss': loss_G,
                'D_loss': loss_D,
            }, './Model/' + save_name + "/" + save_name + '_NetD_' + str(epoch + present_epoch) + '.pth')

        if hasSaved:
            save.append("V")
        else:
            save.append(" ")

        loss_data = pd.DataFrame()
        loss_data['train_G'] = train_losses_G
        loss_data['train_D'] = train_losses_D
        loss_data['test_G'] = test_losses_G
        loss_data['test_D'] = test_losses_D
        loss_data['save'] = save
        loss_data.to_csv('./Model/' + save_name + "/" + save_name + '_loss.csv', index=False)

        plt.plot(train_losses_G, label='train_G')
        plt.plot(train_losses_D, label='train_D')
        plt.plot(test_losses_G, label='test_G')
        plt.plot(test_losses_D, label='test_D')
        plt.legend()
        plt.show()
        # save plot
        plt.savefig('./Model/' + save_name + "/" + save_name + '_loss.png')
        ######################################  Save ######################################

if __name__ == '__main__':
    train()