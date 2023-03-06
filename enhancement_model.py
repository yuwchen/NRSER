


import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import *
import torch.nn.functional as F


class SpeechEnhancement(nn.Module):
    def __init__(self, se_model, n_fft, hop=100):
        super(SpeechEnhancement, self).__init__()
        self.se_model = se_model
        self.n_fft = n_fft
        self.hop = hop
        
    def forward(self, wav):

        wav_spec, c, original_shape =  wav_conversion(wav, self.n_fft, self.hop)
        wav_spec = power_compress(wav_spec).permute(0, 1, 3, 2)

        est_real, est_imag = self.se_model(wav_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)

        if torch.cuda.is_available():
            est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(), onesided=True)
        else:
            est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cpu(), onesided=True)

        est_audio = est_audio / c
        est_audio = torch.reshape(est_audio, original_shape)
        
        return est_audio