import librosa
import torch

# from util.stft import STFT
from Uformer_ofigin.trans import STFT, iSTFT, MelTransform, inv_MelTransform
EPSILON = torch.finfo(torch.float32).eps

def full_band_no_truncation(model, device, inference_args, noisy_ds):
    """
    extract full_band spectra for inference, without truncation.
    """
    n_fft = inference_args["n_fft"]
    hop_length = inference_args["hop_length"]
    win_length = inference_args["win_length"]
    # stft = STFT(win_size=640, hop_size=320)

    noisy_ds = torch.tensor(noisy_ds, device=device)

    output, output_cplx = model(noisy_ds)  #
    enhanced_wav = output.detach().cpu().numpy().reshape(-1)

    return enhanced_wav
