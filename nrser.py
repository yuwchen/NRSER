import os
import argparse
import fairseq
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import *
from SEmodels import generator
import torch.nn.functional as F
from dataloader import MyDataset, MyNoiseDataset
from emotion_model_s import EmotionPredictor
from snr_model import SNRLevelDetection
from enhancement_model import SpeechEnhancement

random.seed(1984)
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='./data',  type=str, help='Path of your DATA/ directory')
    parser.add_argument('--txtfiledir', default='./txtfile',  type=str, help='Path of your training and validation list directory')
    parser.add_argument('--fairseq_base_model', default='../pronunciation/fairseq/hubert_base_ls960.pt', type=str, help='Path to pretrained fairseq base model')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--pretrained_model_emotion', default='', type=str, required=False, help='Path to your Emotion model directory')
    parser.add_argument('--pretrained_model_snr', default='', type=str, required=False, help='Path to your SNR-level detection model directory')
    parser.add_argument('--cmgan_model_path', default='../CMGAN-main/src/best_ckpt/ckpt', type=str)

    args = parser.parse_args()

    cp_path = args.fairseq_base_model
    datadir = args.datadir
    txtfiledir = args.txtfiledir
    my_checkpoint_dir = args.finetune_from_checkpoint
    SE_model_path = args.cmgan_model_path
    pretrained_model_emotion = args.pretrained_model_emotion
    pretrained_model_snr = args.pretrained_model_snr
    ckptdir = os.path.basename(pretrained_model_emotion)+'-'+os.path.basename(pretrained_model_snr)


    if not os.path.exists(ckptdir):
        os.makedirs(os.path.join(ckptdir,'SE'))
        os.makedirs(os.path.join(ckptdir,'EMO'))
        os.makedirs(os.path.join(ckptdir,'SNR'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    wavdir = os.path.join(datadir, '')
    trainlist = os.path.join(txtfiledir, 'emotion_train.txt')
    validlist = os.path.join(txtfiledir, 'emotion_val.txt')


    trainset = MyDataset(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)
    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=8, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    trainlist_n = os.path.join(txtfiledir, 'noise_train.txt')
    validlist_n = os.path.join(txtfiledir, 'noise_val.txt')

    trainset_n = MyNoiseDataset(wavdir, trainlist_n)
    trainloader_n = DataLoader(trainset_n, batch_size=32, shuffle=True, num_workers=2, collate_fn=trainset_n.collate_fn)
    validset_n = MyNoiseDataset(wavdir, validlist_n)
    validloader_n = DataLoader(validset_n, batch_size=32, shuffle=True, num_workers=2, collate_fn=validset_n.collate_fn)

    SSL_OUT_DIM = 768
    N_FFT = 400
    HOP = 100

    if torch.cuda.is_available():
        se_model = generator.TSCNet(num_channel=64, num_features=N_FFT//2+1).cuda()
        se_model.load_state_dict(torch.load(SE_model_path))
    else:
        se_model = generator.TSCNet(num_channel=64, num_features=N_FFT//2+1).cpu()
        se_model.load_state_dict(torch.load(SE_model_path, map_location=torch.device('cpu')))

    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]

    #no need to train the se model, just save the original parameters
    se_net = SpeechEnhancement(se_model, N_FFT, HOP)
    se_net = se_net.to(device)
    torch.save(se_net.state_dict(), os.path.join(ckptdir,'SE','best'))

    emo_net = EmotionPredictor(ssl_model, SSL_OUT_DIM)
    # load pretrained emotion detection model
    emo_net.load_state_dict(torch.load(os.path.join(pretrained_model_emotion,'EMO'+os.sep+'best')))
    emo_net = emo_net.to(device)

    snr_net =  SNRLevelDetection(N_FFT, HOP)
    snr_net = snr_net.to(device)
    #load pretrained noise detection model
    snr_net.load_state_dict(torch.load(os.path.join(pretrained_model_snr,'SNR'+os.sep+'best')))

    if my_checkpoint_dir != None:  ## do (further) finetuning
        snr_net.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'SNR','best')))
        se_net.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'SE','best')))
        emo_net.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'EMO','best')))

    #only train finetune emotion and noise detection model
    criterion_cross = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    optimizer_emo = optim.SGD( emo_net.parameters(), lr=0.0001, momentum=0.9)
    optimizer_noise = optim.SGD( snr_net.parameters(), lr=0.0001, momentum=0.9)
    relu = nn.ReLU()
    
    PREV_VAL_LOSS=9999999999
    orig_patience=2
    patience=orig_patience
    
    for epoch in range(1,1001):
        STEPS=0
        emo_net.train()
        snr_net.train()
        running_loss = 0.0

        for i, data in enumerate(tqdm(trainloader), 0):

            inputs, inputs_en, scores_A, scores_V, scores_D, scores_C, _ = data
            inputs = inputs.to(device)
            inputs_en = inputs_en.to(device)
            scores_A = scores_A.to(device)
            scores_V = scores_V.to(device)
            scores_D = scores_D.to(device)
            scores_C = scores_C.to(device)

            inputs = inputs.squeeze(1)  
            inputs_en = inputs_en.squeeze(1)  
            optimizer_noise.zero_grad()
            optimizer_emo.zero_grad()

            S = snr_net(inputs, inputs_en)        
            snr_level_score = torch.clamp(S, min=0., max=1.)
            emo_input = (inputs_en*(1-snr_level_score) + inputs*snr_level_score)
            output_A, output_V, output_D, output_C = emo_net(emo_input)


            loss_A = ccc_loss(output_A, scores_A)
            loss_V = ccc_loss(output_V, scores_V)
            loss_D = ccc_loss(output_D, scores_D)
            loss_C = criterion_cross(output_C, scores_C)

            loss = loss_C + 1 - (loss_A + loss_V + loss_D )/3

            loss.backward()
            optimizer_emo.step()
            STEPS += 1
            running_loss += loss.item()

        for i, data in enumerate(tqdm(trainloader_n), 0):
            inputs, inputs_en, scores_level, _ = data
            inputs = inputs.to(device)
            inputs_en = inputs_en.to(device)
            scores_level = scores_level.to(device)
            inputs = inputs.squeeze(1)  
            inputs_en = inputs_en.squeeze(1)  
            optimizer_noise.zero_grad()
            noise_level = snr_net(inputs, inputs_en)
            loss = criterion_mse(noise_level, scores_level)
        
            loss.backward()
            optimizer_noise.step()
            STEPS += 1
            running_loss += loss.item()
        
        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))

        ## validation
        VALSTEPS=0
        epoch_val_loss = 0.0

        emo_net.eval()
        snr_net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.memory_allocated()
            torch.cuda.synchronize() 

        for i, data in enumerate(validloader, 0):
            VALSTEPS+=1
            inputs, inputs_en,  scores_A, scores_V, scores_D, scores_C, _ = data
            inputs = inputs.to(device)
            inputs_en = inputs_en.to(device)
            scores_A = scores_A.to(device)
            scores_V = scores_V.to(device)
            scores_D = scores_D.to(device)
            scores_C = scores_C.to(device)
            inputs = inputs.squeeze(1)
            inputs_en = inputs_en.squeeze(1)

            with torch.no_grad(): 
                S = snr_net(inputs, inputs_en)   
                snr_level_score = torch.clamp(S, min=0., max=1.)
                emo_input = (inputs_en*(1-snr_level_score) + inputs*(snr_level_score))
                output_A, output_V, output_D, output_C = emo_net(emo_input)
                loss_A = ccc_loss(output_A, scores_A)
                loss_V = ccc_loss(output_V, scores_V)
                loss_D = ccc_loss(output_D, scores_D)
                loss_C = criterion_cross(output_C, scores_C)

                loss = loss_C + 1 - (loss_A + loss_V + loss_D )/3
                
                epoch_val_loss += loss.item()
        
        for i, data in enumerate(validloader_n, 0):
            VALSTEPS+=1
            inputs, inputs_en,  scores_level, wavname = data
            inputs = inputs.to(device)
            inputs_en = inputs_en.to(device)
            scores_level = scores_level.to(device)
            inputs = inputs.squeeze(1)
            inputs_en = inputs_en.squeeze(1)

            with torch.no_grad(): 
                noise_level = snr_net(inputs, inputs_en)
                loss = criterion_mse(noise_level, scores_level)
                epoch_val_loss += loss.item()

        avg_val_loss=epoch_val_loss/VALSTEPS    
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        if avg_val_loss < PREV_VAL_LOSS:
            print('Loss has decreased')
            PREV_VAL_LOSS=avg_val_loss
            torch.save(emo_net.state_dict(), os.path.join(ckptdir,'EMO','best'))
            torch.save(snr_net.state_dict(), os.path.join(ckptdir,'SNR','best'))
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
        
    print('Finished Fine-tuning')

if __name__ == '__main__':
    main()
