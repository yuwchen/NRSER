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
from dataloader import MyNoiseDataset

random.seed(1984)


class SNRLevelDetection(nn.Module):
    def __init__(self, n_fft, hop):
        super(SNRLevelDetection, self).__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.linear = nn.Linear(201*2,1)

        
    def forward(self, wav, wav_en):
        wav_spec, _, _ =  wav_conversion(wav, self.n_fft, self.hop)
        wav_spec = wav_spec.permute(0, 2, 1, 3)

        wav_spec_en, _, _ =  wav_conversion(wav_en, self.n_fft, self.hop)
        wav_spec_en = wav_spec_en.permute(0, 2, 1, 3)
        # output shape (N, L, H_in, real&image)

        similarity = torch.cosine_similarity(wav_spec, wav_spec_en, dim=1)
        # output shape (N, H_in, real&image)
        
        cosine = torch.mean(similarity, dim=1)
        similarity = torch.reshape(similarity, (-1, 201*2))
        output = self.linear(similarity)
        
        return output

    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='./data',  type=str, help='Path of your DATA/ directory')
    parser.add_argument('--txtfiledir', default='./txtfile',  type=str, help='Path of your DATA/ directory')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='noise_model', help='Output directory for your trained checkpoints')

    args = parser.parse_args()

    datadir = args.datadir
    ckptdir = args.outdir
    txtfiledir = args.txtfiledir
    my_checkpoint_dir = args.finetune_from_checkpoint

    if not os.path.exists(ckptdir):
        os.makedirs(os.path.join(ckptdir,'NOISE'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    wavdir = os.path.join(datadir, '')
    trainlist = os.path.join(txtfiledir, 'noise_train.txt')
    validlist = os.path.join(txtfiledir, 'noise_val.txt')

    N_FFT = 400
    HOP = 100

    trainset = MyNoiseDataset(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)
    validset = MyNoiseDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=32, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    noise_net =  SNRLevelDetection(N_FFT, HOP)
    noise_net = noise_net.to(device)

    if my_checkpoint_dir != None:  ## do (further) finetuning
        noise_net.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'NOISE','best')))
    
    #loss_criterion = nn.L1Loss()
    loss_criterion = nn.MSELoss()

    optimizer_noise = optim.SGD( noise_net.parameters(), lr=0.0001, momentum=0.9)

    PREV_VAL_LOSS=9999999999
    orig_patience=5
    patience=orig_patience

    for epoch in range(1,1001):
        STEPS=0
        noise_net.train()
        running_loss = 0.0

        for i, data in enumerate(tqdm(trainloader), 0):

            inputs, inputs_en, scores_level, _ = data
            inputs = inputs.to(device)
            inputs_en = inputs_en.to(device)
            scores_level = scores_level.to(device)
        
            inputs = inputs.squeeze(1)  
            inputs_en = inputs_en.squeeze(1)  
            optimizer_noise.zero_grad()

            noise_level = noise_net(inputs, inputs_en)
            loss = loss_criterion(noise_level, scores_level)
        
            loss.backward()
            optimizer_noise.step()
            STEPS += 1
            running_loss += loss.item()

        
        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))

        ## validation
        VALSTEPS=0
        epoch_val_loss = 0.0
        noise_net.eval()
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
                noise_level = noise_net(inputs, inputs_en)
                loss = loss_criterion(noise_level, scores_level)
                epoch_val_loss += loss.item()

        avg_val_loss=epoch_val_loss/VALSTEPS    
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        if avg_val_loss < PREV_VAL_LOSS:
            print('Loss has decreased')
            PREV_VAL_LOSS=avg_val_loss
            torch.save(noise_net.state_dict(), os.path.join(ckptdir,'NOISE','best'))
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
        
    print('Finished Training of Noise Model')

if __name__ == '__main__':
    main()
