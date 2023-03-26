import numpy as np
from models import generator
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
import argparse
from tqdm import tqdm
import math
import time

def creat_dir(directory):
      if not os.path.exists(directory):
            os.makedirs(directory)

@torch.no_grad()
def enhance_one_track(model, audio_path, saved_dir, cut_len, n_fft=400, hop=100, save_tracks=False):
    name = os.path.split(audio_path)[-1]
    noisy, sr = torchaudio.load(audio_path)
    if sr!=16000:
        transform = torchaudio.transforms.Resample(sr, 16000)
        noisy = transform(noisy)
        sr = 16000

    assert sr == 16000
    seg_length = 2*sr
    wav_len = noisy.size(-1)
    num_of_seg = math.ceil(wav_len / seg_length) 
    amount_to_pad = num_of_seg*seg_length - wav_len
    noisy = torch.nn.functional.pad(noisy, (0, amount_to_pad), 'constant', 0)
    enhanced_wav = np.zeros(noisy.shape)

    noisy = noisy.cpu()
    #noisy = noisy.cuda() 
    enhanced_wav = enhance_one_seg(noisy, model, cut_len, n_fft=400, hop=100)
    enhanced_wav  = np.reshape(enhanced_wav,(-1,))
    enhanced_wav = enhanced_wav[:wav_len]
    saved_path = os.path.join(saved_dir, name)
    sf.write(saved_path, enhanced_wav, sr)

@torch.no_grad()
def enhance_one_seg(noisy, model, cut_len, n_fft=400, hop=100):
    
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len/cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))

    #noisy_spec = torch.stft(noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True)
    noisy_spec = torch.stft(noisy, n_fft, hop, window=torch.hamming_window(n_fft).cpu(), onesided=True)
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    est_real, est_imag = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    #est_audio = torch.istft(est_spec_uncompress, n_fft, hop, window=torch.hamming_window(n_fft).cuda(),
    #                        onesided=True)
    est_audio = torch.istft(est_spec_uncompress, n_fft, hop, window=torch.hamming_window(n_fft).cpu(),
                            onesided=True)
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    assert len(est_audio) == length

    return est_audio


def evaluation(model_path, noisy_dir, save_tracks, saved_dir):
    n_fft = 400
    #model = generator.TSCNet(num_channel=64, num_features=n_fft//2+1).cuda()
    model = generator.TSCNet(num_channel=64, num_features=n_fft//2+1).cpu()
    model.load_state_dict((torch.load(model_path, map_location=torch.device('cpu'))))
    #model.load_state_dict(torch.load(model_path))

    model.eval()

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    for audio in tqdm(audio_list):
        if audio.endswith('.wav'):
            noisy_path = os.path.join(noisy_dir, audio)
            enhance_one_track(model, noisy_path, saved_dir, 16000*2, n_fft, n_fft//4, save_tracks)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='./CMGAN/best_ckpt/ckpt',
                    help="the path where the model is saved")
parser.add_argument("--test_dir", type=str, default='dir to your VCTK-DEMAND test dataset',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")

args = parser.parse_args()


if __name__ == '__main__':
    save_dir = os.path.join('./data',args.test_dir.split(os.sep)[-1]+'_en')
    evaluation(args.model_path, args.test_dir, args.save_tracks, save_dir)
