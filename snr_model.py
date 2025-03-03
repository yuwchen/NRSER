import os
import argparse
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import *
from dataloader_prob import MyNoiseDataset

random.seed(1984)


class SNRLevelDetection(nn.Module):
    def __init__(self, n_fft, hop):
        super(SNRLevelDetection, self).__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.fusion_layer = nn.TransformerEncoderLayer(d_model=2, nhead=1)
        self.cnn = nn.Conv1d(2, 1, kernel_size=1)

        
    def forward(self, wav, wav_en):
        
        max_val = torch.max(abs(wav))
        wav = wav/max_val

        wav_spec, _, _ =  wav_conversion(wav, self.n_fft, self.hop)
        wav_spec = wav_spec.permute(0, 2, 1, 3)

        
        wav_spec_en, _, _ =  wav_conversion(wav_en, self.n_fft, self.hop)
        wav_spec_en = wav_spec_en.permute(0, 2, 1, 3)
        # output shape (N, L, H_in, real&image)

        similarity = torch.cosine_similarity(wav_spec, wav_spec_en, dim=2)
        # output shape (N, H_in, real&image)
        similarity = self.fusion_layer(similarity)
        similarity = similarity.transpose(1, 2)

        similarity = self.cnn(similarity)

        output = torch.mean(similarity, dim=2)
        #similarity = torch.reshape(similarity, (-1, 201*2))
        return output

    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='./data',  type=str, help='Path of your DATA/ directory')
    parser.add_argument('--txtfiledir', default='./txtfile',  type=str, help='Path of your DATA/ directory')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='snr_model', help='Output directory for your trained checkpoints')

    args = parser.parse_args()

    datadir = args.datadir
    ckptdir = args.outdir
    txtfiledir = args.txtfiledir
    my_checkpoint_dir = args.finetune_from_checkpoint

    if not os.path.exists(ckptdir):
        os.makedirs(os.path.join(ckptdir,'SNR'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    wavdir = os.path.join(datadir, '')
    trainlist = os.path.join(txtfiledir, 'snr-audioset-train-80-msp1_11-train-clean.txt')
    validlist = os.path.join(txtfiledir, 'snr-audioset-val-msp1_11-test2-clean.txt')

    N_FFT = 400
    HOP = 100

    trainset = MyNoiseDataset(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)
    validset = MyNoiseDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=32, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    snr_net =  SNRLevelDetection(N_FFT, HOP)
    snr_net = snr_net.to(device)

    if my_checkpoint_dir != None:  ## do (further) finetuning
        snr_net.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'NOISE','best')))
    
    #loss_criterion = nn.L1Loss()
    loss_criterion = nn.MSELoss()

    optimizer_noise = optim.SGD( snr_net.parameters(), lr=0.0001, momentum=0.9)

    PREV_VAL_LOSS=9999999999
    orig_patience=2
    patience=orig_patience

    for epoch in range(1,1001):
        STEPS=0
        snr_net.train()
        running_loss = 0.0

        for i, data in enumerate(tqdm(trainloader), 0):

            inputs, inputs_en, scores_level, wavname = data
            inputs = inputs.to(device)
            inputs_en = inputs_en.to(device)
            scores_level = scores_level.to(device)
            
            inputs = inputs.squeeze(1)  
            inputs_en = inputs_en.squeeze(1)  
            optimizer_noise.zero_grad()
            print(inputs.shape, inputs_en.shape)
            noise_level = snr_net(inputs, inputs_en)
            loss = loss_criterion(noise_level, scores_level)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            optimizer_noise.step()
            STEPS += 1
            running_loss += loss.item()

        
        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))

        ## validation
        VALSTEPS=0
        epoch_val_loss = 0.0
        snr_net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.memory_allocated()
            torch.cuda.synchronize() 
    
        for i, data in enumerate(validloader, 0):
            VALSTEPS+=1
            inputs, inputs_en,  scores_level, wavname = data
            inputs = inputs.to(device)
            inputs_en = inputs_en.to(device)
            scores_level = scores_level.to(device)
            inputs = inputs.squeeze(1)
            inputs_en = inputs_en.squeeze(1)

            with torch.no_grad(): 
                noise_level = snr_net(inputs, inputs_en)
                loss = loss_criterion(noise_level, scores_level)
                if torch.isnan(loss)  or torch.isinf(loss): 
                    continue
                epoch_val_loss += loss.item()

        avg_val_loss=epoch_val_loss/VALSTEPS    
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        if avg_val_loss < PREV_VAL_LOSS:
            print('Loss has decreased')
            PREV_VAL_LOSS=avg_val_loss
            torch.save(snr_net.state_dict(), os.path.join(ckptdir,'NOISE','best'))
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
        
    print('Finished Training of Noise Model')

if __name__ == '__main__':
    main()
