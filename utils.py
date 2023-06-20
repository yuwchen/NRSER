import torch
import torch.nn as nn

def ccc_loss(y_true, y_pred):
 
    merge = torch.cat((y_true.unsqueeze(1).T, y_pred.unsqueeze(1).T),0)
    cor = torch.corrcoef(merge)[0][1]
 
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
 
    var_true = torch.var(y_true, unbiased=False)
    var_pred = torch.var(y_pred, unbiased=False)
 
    sd_true = torch.std(y_true, unbiased=False)
    sd_pred = torch.std(y_pred, unbiased=False)
 
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator

def wav_conversion(wav, n_fft, hop):
    original_shape = wav.shape
    c = torch.sqrt(wav.size(-1) / torch.sum((wav ** 2.0), dim=-1))
    wav  = torch.transpose(wav, 0, 1)
    wav  = torch.transpose(wav * c, 0, 1)
    if torch.cuda.is_available():
        wav_spec = torch.stft(wav, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True)
    else:
        wav_spec = torch.stft(wav, n_fft, hop, window=torch.hamming_window(n_fft).cpu(), onesided=True)
    return  wav_spec, c, original_shape
    
def kaiming_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1./0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
