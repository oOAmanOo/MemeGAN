import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import BCELoss, CrossEntropyLoss
from extractor import addImagePath, textExtraction, imageExtraction, textExtractReverse
import matplotlib.pyplot as plt

def train(args):
    #############################################################################################
    # 1. Load data
    #############################################################################################

    if args.data == 'Oxford_HIC':
        dirPath = '../Data/Oxford_HIC/oxford_hic_data.csv'
        imgPath = '../Data/Oxford_HIC/oxford_img/'
    else:
        dirPath = '../Data/Instagram/Filter_' + 'wendys' + '.csv'
        imgPath = '../Data/Instagram/' + 'wendys' + '_img/'
    # load data
    data = pd.read_csv(dirPath)
    data = addImagePath(data, imgPath)
    # split data
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train_text = textExtraction(train['caption'].tolist())
    train_image = imageExtraction(train['image_id'])
    train_funny_score = torch.tensor(train['funny_score'].to_numpy())
    test_text = textExtraction(test['caption'])
    test_image = imageExtraction(test['image_id'])
    test_funny_score = torch.tensor(test['funny_score'].to_numpy())

    train_dataset = torch.utils.data.TensorDataset(train_text, train_image, train_funny_score)
    test_dataset = torch.utils.data.TensorDataset(test_text, test_image, test_funny_score)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    #############################################################################################
    # 2. Define model
    #############################################################################################
    gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", device_map="auto", torch_dtype=torch.bfloat16)

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            # self attention
            self.selfAttentionMultihead = nn.MultiheadAttention(768, 1)
            self.selfAttentionLayerNorm = nn.LayerNorm(768)
            self.selfAttentionLinear = nn.Linear(768, 768)
            self.selfAttentionLayerNorm2 = nn.LayerNorm(768)

            # multihead attention
            self.multiheadAttentionMultihead = nn.MultiheadAttention(768, 8)
            self.multiheadAttentionLinear = nn.Linear(768, 768)
            self.multiheadAttentionLayerNorm = nn.LayerNorm(768)

            # co-attention text
            self.coAttentionTextMultihead = nn.MultiheadAttention(768, 1)
            self.coAttentionTextLinear = nn.Linear(768, 768)
            self.coAttentionTextLayerNorm = nn.LayerNorm(768)

            # co-attention image
            self.coAttentionImageMultihead = nn.MultiheadAttention(768, 1)
            self.coAttentionImageLinear = nn.Linear(768, 768)
            self.coAttentionImageLayerNorm = nn.LayerNorm(768)

            # feed forward
            self.feedForwardLinear = nn.Linear(768, 768)
            self.feedForwardLayerNorm = nn.LayerNorm(768)

            # gemma
            self.gemmaLinearBefore = nn.Linear(768, 50265)
            self.gemmaSoftmax = nn.Softmax(dim=2)
            self.gemma = nn.Sequential(*list(gemma.children())[:-1])
            self.gemmaLm_head = nn.Sequential(*list(gemma.children())[1:])

            # funny score
            self.FunnyScorelinear1 = nn.Linear(768, 1)
            self.FunnyScorelinear2 = nn.Linear(64, 1)

        def gemmaGenerate(self, x):
            with torch.no_grad():
                x = self.gemmaLinearBefore(x)
                x = self.gemmaSoftmax(x)
                # get max value of each row, total 32*64
                top_k_values, top_k_indices = torch.topk(x, 1, dim=2, largest=True)
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
            # self attention module
            self_out = self.selfAttentionMultihead(image, image, image)[0]
            self_out = self.selfAttentionLinear(self_out)
            self_out = self.selfAttentionLayerNorm(self_out + image)

            # multihead attention module
            multi_out = self.multiheadAttentionMultihead(text, text, text)[0]
            multi_out = self.multiheadAttentionLinear(multi_out)
            multi_out = self.multiheadAttentionLayerNorm(multi_out + text)

            # co-attention image module
            visual_attending_textual = self.coAttentionTextMultihead(self_out, multi_out, multi_out)[0]
            visual_attending_textual = self.coAttentionTextLinear(visual_attending_textual)
            visual_attending_textual = self.coAttentionTextLayerNorm(visual_attending_textual + self_out)

            # co-attention text module
            textual_attending_visual = self.coAttentionTextMultihead(multi_out, self_out, self_out)[0]
            textual_attending_visual = self.coAttentionTextLinear(textual_attending_visual)
            textual_attending_visual = self.coAttentionTextLayerNorm(textual_attending_visual + multi_out)
            ###############################################################

            # feature fusion
            feature_fusion = visual_attending_textual + textual_attending_visual
            feature_fusion = self.feedForwardLinear(feature_fusion)
            feature_fusion = self.feedForwardLayerNorm(feature_fusion + feature_fusion)
            feature_fusion = feature_fusion.squeeze(-1)
            feature_fusion = feature_fusion.transpose(0, 1)

            ####################### gemma  generate #######################
            last_hidden_state = self.gemmaGenerate(feature_fusion)
            output_text = self.gemmaLm_head(last_hidden_state)
            ###############################################################

            ######################### funny score #########################
            output_funny_score = self.FunnyScorelinear1(feature_fusion).squeeze(-1)
            output_funny_score = self.FunnyScorelinear2(output_funny_score).squeeze(-1)
            ###############################################################

            return output_text, output_funny_score

        def generate(self, image, max_length=100):
            generated_tokens = []
            generated_tokens.append(2)  # <bos> = 2
            text = torch.zeros_like(image).to(device)
            text = text.transpose(0, 1)
            image = image.transpose(0, 1)
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

            # 有時後空格會失效，所以手動插入空格
            def insert_zeros(list):
                # <pad> = 0, <eos> = 1, <bos> = 2, <unk> = 3, <mask> = 4, <2mass> = 5, [@BOS@] = 6
                zeros = [0] * (2 * len(list) - 1)
                zeros[::2] = list
                return zeros

            with torch.no_grad():
                for _ in range(max_length):
                    # self attention module
                    self_out = self.selfAttentionMultihead(image, image, image)[0]
                    self_out = self.selfAttentionLinear(self_out)
                    self_out = self.selfAttentionLayerNorm(self_out + image)
                    # multihead attention module
                    multi_out = self.multiheadAttentionMultihead(text, text, text)[0]
                    multi_out = self.multiheadAttentionLinear(multi_out)
                    multi_out = self.multiheadAttentionLayerNorm(multi_out + text)
                    # co-attention image module
                    visual_attending_textual = self.coAttentionTextMultihead(self_out, multi_out, multi_out)[0]
                    visual_attending_textual = self.coAttentionTextLinear(visual_attending_textual)
                    visual_attending_textual = self.coAttentionTextLayerNorm(visual_attending_textual + self_out)

                    # co-attention text module
                    textual_attending_visual = self.coAttentionTextMultihead(multi_out, self_out, self_out)[0]
                    textual_attending_visual = self.coAttentionTextLinear(textual_attending_visual)
                    textual_attending_visual = self.coAttentionTextLayerNorm(textual_attending_visual + multi_out)

                    # feature fusion
                    feature_fusion = visual_attending_textual + textual_attending_visual
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

                    next_token_logits = output_text[:, -1, :]
                    next_token_probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.argmax(next_token_probs, dim=-1).item()
                    generated_tokens.append(next_token_id)

                    generated_caption = insert_zeros(generated_tokens)
                    generated_caption = tokenizer.decode(generated_caption, skip_special_tokens=False)
                    generated_caption = generated_caption.replace("<pad>", " ").replace("  ", " ").split()
                    generated_caption = [word for word in generated_caption if word[0] != "<"]
                    generated_caption = " ".join(generated_caption)

                    if next_token_id == 1:  # <eos> = 1
                        break
                    else:
                        text = textExtraction([generated_caption]).to(device)
                        text = text.transpose(0, 1)

            return generated_caption, output_funny_score

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            # Generator
            self.g_linearFake1 = nn.Linear(256000, 768)
            self.g_linearFake2 = nn.Linear(256, 64)
            self.g_con_mlp1 = nn.Linear(768, 1)
            self.g_con_mlp2 = nn.Linear(128, 1)
            self.g_unc_mlp1 = nn.Linear(768, 1)
            self.g_unc_mlp2 = nn.Linear(64, 1)
            # Discriminator
            self.d_linearFake1 = nn.Linear(256000, 768)
            self.d_linearFake2 = nn.Linear(256, 64)
            self.d_con_mlp1_r = nn.Linear(768, 1)
            self.d_con_mlp2_r = nn.Linear(256, 1)
            self.d_con_mlp1_g = nn.Linear(768, 1)
            self.d_con_mlp2_g = nn.Linear(128, 1)
            self.d_con_mlp1_m = nn.Linear(768, 1)
            self.d_con_mlp2_m = nn.Linear(128, 1)
            self.d_unc_mlp1_r = nn.Linear(768, 1)
            self.d_unc_mlp2_r = nn.Linear(64, 1)
            self.d_unc_mlp1_g = nn.Linear(768, 1)
            self.d_unc_mlp2_g = nn.Linear(64, 1)
            self.d_unc_mlp1_m = nn.Linear(768, 1)
            self.d_unc_mlp2_m = nn.Linear(64, 1)

        def forward(self, real_text, fake_text, image):
            # real_text = [batch_size, 64, 768]
            # fake_text = [batch_size, 256, 256000]
            # image = [batch_size, 64, 768]
            g_fake_text = self.g_linearFake1(fake_text).transpose(1, 2)
            g_fake_text = self.g_linearFake2(g_fake_text).transpose(1, 2)
            d_fake_text = self.d_linearFake1(fake_text).transpose(1, 2)
            d_fake_text = self.d_linearFake2(d_fake_text).transpose(1, 2)
            mismatched_text = torch.roll(real_text, 1, 0)

            # conditional (contrastive)
            C_r = torch.cat((real_text, image), dim=1)
            g_C_g = torch.cat((g_fake_text, image), dim=1)
            d_C_g = torch.cat((d_fake_text, image), dim=1)
            C_m = torch.cat((mismatched_text, image), dim=1)
            # contrastive discriminator
            d_C_r = torch.cat((C_r, d_C_g), dim=1)

            ########################## Generator ##########################
            g_C_g = self.g_con_mlp1(g_C_g).squeeze(-1)
            g_C_g = self.g_con_mlp2(g_C_g).squeeze(-1).unsqueeze(0)  # (32x320 and 128x1)
            ###############################################################

            ######################## Discriminator ########################
            d_C_r = self.d_con_mlp1_r(d_C_r).squeeze(-1)
            d_C_g = self.d_con_mlp1_g(d_C_g).squeeze(-1)
            d_C_m = self.d_con_mlp1_m(C_m).squeeze(-1)
            d_C_r = self.d_con_mlp2_r(d_C_r).squeeze(-1).unsqueeze(0)
            d_C_g = self.d_con_mlp2_g(d_C_g).squeeze(-1).unsqueeze(0)
            d_C_m = self.d_con_mlp2_m(d_C_m).squeeze(-1).unsqueeze(0)
            d_con_output = torch.cat((d_C_r, d_C_g, d_C_m), dim=0)
            ###############################################################

            #### unconditional ####
            ########################## Generator ##########################
            g_UC_g = self.g_unc_mlp1(g_fake_text).squeeze(-1)
            g_UC_g = self.g_unc_mlp2(g_UC_g).squeeze(-1).unsqueeze(0)
            g_output = torch.cat((g_C_g, g_UC_g), dim=0)
            ###############################################################

            ######################## Discriminator ########################
            d_UC_r = self.d_unc_mlp1_r(real_text).squeeze(-1)
            d_UC_g = self.d_unc_mlp1_g(d_fake_text).squeeze(-1)
            d_UC_m = self.d_unc_mlp1_m(mismatched_text).squeeze(-1)
            d_UC_r = self.d_unc_mlp2_r(d_UC_r).squeeze(-1).unsqueeze(0)
            d_UC_g = self.d_unc_mlp2_g(d_UC_g).squeeze(-1).unsqueeze(0)
            d_UC_m = self.d_unc_mlp2_m(d_UC_m).squeeze(-1).unsqueeze(0)
            d_unc_output = torch.cat((d_UC_r, d_UC_g, d_UC_m), dim=0)
            ###############################################################

            # torch.Size([3, 32, 1])
            return g_output, d_con_output, d_unc_output

    #############################################################################################
    # 3. Train model
    #############################################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NetG = Generator().to(device)
    NetD = Discriminator().to(device)
    optimizer_G = optim.Adam(NetG.parameters(), lr=args.generator_learning_rate)
    optimizer_D = optim.Adam(NetD.parameters(), lr=args.discriminator_learning_rate)
    funnyScoreLoss = nn.MSELoss()

    def generatorLoss(generator_logits):
        m = nn.Sigmoid()
        result_fake = (torch.zeros(generator_logits[1].shape[0])).to(device)
        unc_loss = BCELoss()(m(generator_logits[1]), result_fake)
        con_loss = BCELoss()(m(generator_logits[0]), result_fake)
        loss = con_loss + unc_loss
        return loss

    def discriminatorLoss(uncondition_logits, condition_logits):
        m = nn.Sigmoid()
        result_true = (torch.ones(condition_logits[0].shape[0])).to(device)
        result_fake = (torch.zeros(condition_logits[0].shape[0])).to(device)
        unc_r = BCELoss()(m(condition_logits[0]), result_true)
        unc_f = BCELoss()(m(condition_logits[1]), result_fake)
        unc_m = BCELoss()(m(condition_logits[2]), result_fake)
        con_r = CrossEntropyLoss()(uncondition_logits[0], result_true)
        con_f = CrossEntropyLoss()(uncondition_logits[1], result_fake)
        con_m = CrossEntropyLoss()(uncondition_logits[2], result_fake)
        loss = unc_r + ((unc_f + unc_m) / 2) + con_r + ((con_f + con_m) / 2)
        return loss

    train_losses_FC = []
    train_losses_G = []
    train_losses_D = []
    test_losses_FC = []
    test_losses_G = []
    test_losses_D = []

    if not os.path.exists('./Model/' + args.exp_name):
        os.makedirs('./Model/' + args.exp_name)


    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.num_epochs):
        print("---------------------------------------- epoch " + str(
            epoch + 1) + " ---------------------------------------")
        train_loss_FC = 0
        train_loss_G = 0
        train_loss_D = 0
        test_loss_FC = 0
        test_loss_G = 0
        test_loss_D = 0

        ###################################### Train ######################################
        with tqdm(train_loader, unit="batch") as tepoch:
            for text, image, funny_score in tepoch:
                ######################################################
                # (1) Generate fake caption
                ######################################################
                logits, output_funny_score = NetG(text.to(device).to(torch.float32), image.to(device).to(torch.float32))
                gen_logits, con_logits, unc_logits = NetD(text.to(device).to(torch.float32),
                                                          logits.detach().to(torch.float32),
                                                          image.to(device).to(torch.float32))
                ######################################################
                # (3) Update Discriminator network
                #####################################################
                optimizer_D.zero_grad()
                loss_D = discriminatorLoss(unc_logits, con_logits)
                loss_D.backward(retain_graph=True)
                optimizer_D.step()
                train_loss_D += loss_D.item()
                ######################################################
                # (4) Update Generator network
                ######################################################
                optimizer_G.zero_grad()
                loss_FC = funnyScoreLoss(output_funny_score, funny_score.to(device).to(torch.float32))
                loss_FC.backward(retain_graph=True)
                train_loss_FC += loss_FC.item()
                loss_G = generatorLoss(gen_logits)
                loss_G.backward()
                optimizer_G.step()
                train_loss_G += loss_G.item()
                ######################################################
                tepoch.set_postfix({'FC_loss': train_loss_FC, 'G_loss': train_loss_G, 'D_loss': train_loss_D})
                ######################################################
                # (5) Save the model
                ######################################################
                nameEpoch = args.exp_name + "_" + str(epoch + 1)
                torch.save(NetG.state_dict(), './Model/' + args.exp_name + "/" + nameEpoch + 'NetG.pth')
                torch.save(NetD.state_dict(), './Model/' + args.exp_name + "/" + nameEpoch + 'NetD.pth')
                ######################################################
        train_losses_FC.append(train_loss_FC)
        train_losses_G.append(train_loss_G)
        train_losses_D.append(train_loss_D)
        ###################################### Train ######################################

        ######################################  Test ######################################
        with tqdm(test_loader, unit="batch") as tepoch:
            for text, image, funny_score in tepoch:
                # Generator
                logits, output_funny_score = NetG(text.to(device).to(torch.float32), image.to(device).to(torch.float32))
                # Discriminator
                gen_logits, con_logits, unc_logits = NetD(text.to(device).to(torch.float32),
                                                          logits.detach().to(torch.float32),
                                                          image.to(device).to(torch.float32))
                # loss
                loss_FC = funnyScoreLoss(output_funny_score, funny_score.to(device).to(torch.float32))
                loss_G = generatorLoss(gen_logits)
                loss_D = discriminatorLoss(unc_logits, con_logits)
                test_loss_FC += loss_FC.item()
                test_loss_G += loss_G.item()
                test_loss_D += loss_D.item()
                tepoch.set_postfix({'FC_loss': test_loss_FC, 'G_loss': test_loss_G, 'D_loss': test_loss_D})
        test_losses_FC.append(test_loss_FC)
        test_losses_G.append(test_loss_G)
        test_losses_D.append(test_loss_D)
        ######################################  Test ######################################

    plt.plot(train_losses_FC, label='train')
    plt.plot(train_losses_G, label='train')
    plt.plot(train_losses_D, label='train')
    plt.plot(test_losses_FC, label='test')
    plt.plot(test_losses_G, label='test')
    plt.plot(test_losses_D, label='test')
    plt.legend()
    plt.show()
    # save plot
    plt.savefig('./Model/' + args.exp_name + "/" + args.exp_name + '_loss.png')

    # to csv
    train_losses = pd.DataFrame({'train_FC': train_losses_FC, 'train_G': train_losses_G, 'train_D': train_losses_D})
    test_losses = pd.DataFrame({'test_FC': test_losses_FC, 'test_G': test_losses_G, 'test_D': test_losses_D})
    train_losses.to_csv('./Model/' + args.exp_name + "/" + args.exp_name + '_train_losses.csv', index=False)
    test_losses.to_csv('./Model/' + args.exp_name + "/" + args.exp_name + '_test_losses.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-glr','--learning-rate' , type=float, default=5e-5, dest='generator_learning_rate',
                        help='Generator learning rate during training')
    parser.add_argument('-dlr', '--learning-rate', type=float, default=5e-5, dest='discriminator_learning_rate',
                        help='Discrimiator learning rate during training')
    # parser.add_argument('-gwd', '--weight-decay', type=float, default=0.1, dest='generator_weight_decay',
    #                     help='weight decay during training')
    # parser.add_argument('-dwd', '--weight-decay', type=float, default=0.1, dest='discriminator_weight_decay',
    #                     help='weight decay during training')
    parser.add_argument('-e', '--num-epochs', type=int, default=30, dest='num_epochs',
                        help='number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=32, dest='batch_size',
                        help='batch size during training')
    parser.add_argument('-data', type=str, default='Oxford_HIC', dest='data',
                        help='image directory where Rico dataset is stored')
    parser.add_argument('-name', '--exp-name',type=str, default='test', dest='exp_name',
                        help='experiment name')

    args = parser.parse_args()
    train(args)



