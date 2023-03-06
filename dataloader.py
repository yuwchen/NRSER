import os
import torch
import numpy as np
import torchaudio
from torch.utils.data.dataset import Dataset


class MyDataset(Dataset):

    def __init__(self, rootdir, data_list):

        self.label_map = {'A':0, 'S':1, 'H':2, 'U':3, 'F':4, 'D':5, 'C':6, 'N':7, 'O':8, 'X':9}

        self.A_lookup = {}
        self.V_lookup = {}
        self.D_lookup = {}
        self.C_lookup = {}

        f = open(data_list, 'r').read().splitlines()
        wavfiles = []
        for line in f:
            parts = line.split(';')
            wavfile = parts[0]
            category = self.label_map[parts[1].replace(' ','')]
            arousal = float(parts[2].split(':')[1])
            valence = float(parts[3].split(':')[1])
            dominance = float(parts[4].split(':')[1])
            self.A_lookup[wavfile] = arousal
            self.V_lookup[wavfile] = valence
            self.D_lookup[wavfile] = dominance
            self.C_lookup[wavfile] = category

            wavfiles.append(wavfile)

        self.rootdir = rootdir
        self.wavfiles = sorted(wavfiles)
        
    def __getitem__(self, idx):
        wavfile = self.wavfiles[idx]
        wavpath = os.path.join(self.rootdir, wavfile)
        wav = torchaudio.load(wavpath)[0]
        wavdir_en = wavfile.split(os.sep)[0]+'_en'
        wavpath_en = os.path.join(self.rootdir, wavdir_en, os.path.basename(wavfile))
        
        wav_en = torchaudio.load(wavpath_en)[0]

        score_A = self.A_lookup[wavfile]
        score_V = self.V_lookup[wavfile]
        score_D = self.D_lookup[wavfile]
        score_C = self.C_lookup[wavfile]

        return wav, wav_en, score_A, score_V, score_D, score_C, wavfile
    
    def __len__(self):
        return len(self.wavfiles)

    def collate_fn(self, batch):  ## zero padding
        
        wavs, wavs_en, score_A, score_V, score_D, score_C, wavfile = zip(*batch)    
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]

        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        output_wavs = torch.stack(output_wavs, dim=0)

        wavs_en = list(wavs_en)
        max_len = max(wavs_en, key = lambda x : x.shape[1]).shape[1]
        output_wavs_en = []
        for wav in wavs_en:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs_en.append(padded_wav)
        output_wavs_en = torch.stack(output_wavs_en, dim=0)

        scores_A  = torch.stack([torch.tensor(x) for x in list(score_A)], dim=0)
        scores_V  = torch.stack([torch.tensor(x) for x in list(score_V)], dim=0)
        scores_D  = torch.stack([torch.tensor(x) for x in list(score_D)], dim=0)
        scores_C  = torch.stack([torch.tensor(x) for x in list(score_C)], dim=0)

        return output_wavs, output_wavs_en, scores_A, scores_V, scores_D, scores_C, wavfile



class MyNoiseDataset(Dataset):

    def __init__(self, wavdir, data_list):

        self.level_lookup = {}
        f = open(data_list, 'r').read().splitlines()
        for line in f:
            parts = line.split(';')
            wavname = parts[0]
            level = float(parts[1])
            self.level_lookup[wavname] = level
        self.wavdir = wavdir
        self.wavnames = sorted(self.level_lookup.keys())

        
    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        wavdir_en = wavname.split("/")[0]+'_en'
        wavpath_en = os.path.join(self.wavdir, wavdir_en, os.path.basename(wavname))
        wav_en = torchaudio.load(wavpath_en)[0]
        score_level = self.level_lookup[wavname]

        return wav, wav_en, score_level, wavname
    
    def __len__(self):
        return len(self.wavnames)

    def collate_fn(self, batch):  ## zero padding
        
        wavs, wavs_en, score_level, wavnames = zip(*batch)
    
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        output_wavs = torch.stack(output_wavs, dim=0)
        
        wavs_en = list(wavs_en)
        max_len = max(wavs_en, key = lambda x : x.shape[1]).shape[1]
        output_wavs_en = []
        for wav in wavs_en:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs_en.append(padded_wav)
        output_wavs_en = torch.stack(output_wavs_en, dim=0)

        scores_level  = torch.stack([torch.tensor(x) for x in list(score_level)], dim=0)
        scores_level = torch.reshape(scores_level,(-1,1))

        return output_wavs, output_wavs_en, scores_level, wavnames
