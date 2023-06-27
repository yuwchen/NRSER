import os
import gc
import math
import argparse
import torch
import torch.nn as nn
import fairseq
import numpy as np
import torchaudio
import soundfile as sf
from tqdm import tqdm
from utils import *
from SEmodels import generator
from snr_model import SNRLevelDetection
from enhancement_model import SpeechEnhancement

gc.collect()
torch.cuda.empty_cache()


def get_filepaths(directory):
      file_paths = []  
      for root, _, files in os.walk(directory):
            for filename in files:
                  filepath = os.path.join(root, filename)
                  if filename.endswith('.wav'):
                        file_paths.append(filepath)  
      return file_paths 


def en_one_track(noisy, sr, se_model, device):
    
    assert sr == 16000
    seg_length = 2*sr
    wav_len = noisy.size(-1)
    num_of_seg = math.ceil(wav_len / seg_length) 
    amount_to_pad = num_of_seg*seg_length - wav_len
    noisy = torch.nn.functional.pad(noisy, (0, amount_to_pad), 'constant', 0)
    enhanced_wav = torch.zeros(noisy.shape).to(device)

    noisy = noisy.cuda()

    for i in range (num_of_seg):
        wav_seg = noisy[:,i*seg_length:(i+1)*seg_length]
        en_seg = se_model(wav_seg)
        enhanced_wav[:,i*seg_length:(i+1)*seg_length] = en_seg
    
    enhanced_wav = enhanced_wav[:,:wav_len]
    return enhanced_wav

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmgan_base_model', type=str, default='../CMGAN-main/src/best_ckpt/ckpt', help='Path to pretrained CMGAN model.')
    parser.add_argument('--datadir', type=str, help='Path of your DATA/ directory')
    parser.add_argument('--ckptdir', type=str, help='Path to pretrained checkpoint.')
    parser.add_argument('--savewav', type=bool, help='Whether to save the enhanced wav')

    args = parser.parse_args()
    
    se_path = args.cmgan_base_model
    my_checkpoint_dir = args.ckptdir
    datadir = args.datadir
    savewav = args.savewav
    
    N_FFT = 400
    HOP=100

    cmgan = generator.TSCNet(num_channel=64, num_features=N_FFT//2+1).cuda()
    cmgan.load_state_dict(torch.load(se_path))

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    snr_model = SNRLevelDetection(N_FFT, HOP).to(device)
    snr_model.eval()
    snr_model.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'SNR'+os.sep+'best')))

    se_model = SpeechEnhancement(cmgan, N_FFT, HOP).to(device)
    se_model.eval()
    se_model.load_state_dict(torch.load(os.path.join(my_checkpoint_dir,'SE'+os.sep+'best')))

    print('Loading data')

    validset = get_filepaths(datadir)
    outfile = my_checkpoint_dir.split("/")[-1]+'-'+datadir.split('/')[-1]+'.txt'

    output_dir = 'Results_snr'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if savewav:
        wav_output_dir = os.path.join('Enhanced_wav',my_checkpoint_dir.split(os.sep)[-1])
        if not os.path.exists(wav_output_dir):
            os.makedirs(wav_output_dir)

    prediction = open(os.path.join(output_dir, outfile), 'w')
    
    print('Starting prediction')
    for filepath in tqdm(validset):
        
        with torch.no_grad():
            filename = filepath.split("/")[-1]
            wav, sr = torchaudio.load(filepath)
            if sr!=16000:
                transform = torchaudio.transforms.Resample(sr, 16000)
                wav = transform(wav)
                sr = 16000
            wav = wav.to(device)

            en_wav = en_one_track(wav, sr, se_model, device)
            if savewav:
                the_wav = en_wav.cpu().detach().numpy()
                the_wav = np.reshape(the_wav,(-1,))
                sf.write(wav_output_dir+os.sep+filename, the_wav, 16000)
                
            en_wav = en_wav.to(device)

            S = snr_model(wav, en_wav)
            snr_level_score = S.cpu().detach().numpy()
            snr_level_score = snr_level_score[0][0]
            torch.cuda.empty_cache()
            output = "{}; S:{}".format(filename, snr_level_score)
            prediction.write(output+'\n')
 


if __name__ == '__main__':
    main()
