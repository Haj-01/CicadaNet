import librosa
import torch
from util.my_stft import STFT


def full_band_no_truncation(model, device, inference_args, noisy):
    """
    extract full_band spectra for inference, without truncation.
    """
    n_fft = inference_args["n_fft"]
    hop_length = inference_args["hop_length"]
    win_length = inference_args["win_length"]
    masking_mode = 'C'

    stft = STFT(win_size=win_length, hop_size=hop_length).to(device)

    #mag
    noisy_mag, noisy_phase = librosa.magphase(librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    noisy_mag = torch.tensor(noisy_mag, device=device)[None, None, :, :]  # [F, T] => [1, 1, F, T]

    # mag mapping
    # enhanced_mag = model(noisy_mag)  # [1, 1, F, T] => [1, 1, F, T]
    # enhanced_mag = enhanced_mag.squeeze(0).squeeze(0).detach().cpu().numpy()  # [1, 1, F, T] => [F, T]
    # enhanced = librosa.istft(enhanced_mag * noisy_phase, hop_length=hop_length, win_length=win_length, length=len(noisy))

    # mag mask
    mask = model(noisy_mag)  # [1, 1, F, T] => [1, 1, F, T]
    enhanced_mag = noisy_mag * mask
    enhanced_mag = enhanced_mag.squeeze(0).squeeze(0).detach().cpu().numpy()  # [1, 1, F, T] => [F, T]
    #
    enhanced = librosa.istft(enhanced_mag * noisy_phase, hop_length=hop_length, win_length=win_length, length=len(noisy))


    #cplx
    # noisy_tensor = torch.tensor(noisy, device=device)[None, :]  # [T] => [1, T]
    # noisy_real, noisy_imag = stft.stft(noisy_tensor)  # B x T x F
    # noisy_real = noisy_real.permute(0, 2, 1)  # B x F x T
    # noisy_imag = noisy_imag.permute(0, 2, 1)
    # # noisy_phase = torch.atan2(noisy_imag, noisy_real)
    # noisy_cplx = torch.stack([noisy_real, noisy_imag], 1)  # N x 2 x F x T
    #
    # out = model(noisy_cplx)  # [B, 2, F, T]

    #cplx mask
    # mask_real = out[:, 0]
    # mask_imag = out[:, 1]
    #
    # if masking_mode == 'C':
    #     enh_real, enh_imag = noisy_real * mask_real - noisy_imag * mask_imag, noisy_real * mask_imag + noisy_imag * mask_real
    # elif masking_mode == 'R':
    #     enh_real, enh_imag = noisy_real * mask_real, noisy_imag * mask_imag
    #
    # enhanced_cplx = torch.stack([enh_real, enh_imag], 1)  # N x 2 x F x T
    # enhanced_cplx = torch.transpose(enhanced_cplx, 2, 3)  # B C T F
    # enhanced = stft.istft(enhanced_cplx)
    # enhanced = enhanced.squeeze(0).detach().cpu().numpy()  # [1, T] => [T]

    # enh_mag = torch.sqrt(enh_real ** 2 + enh_imag ** 2 + 1e-8)  # B F T
    # enh_phase = torch.atan2(enh_imag, enh_real)
    # enhanced_mag = enh_mag.squeeze(0).detach().cpu().numpy()  # [1, F, T] => [F, T]
    # enhanced_phase = enh_phase.squeeze(0).detach().cpu().numpy()  # [1, F, T] => [F, T]
    # enhanced = librosa.istft(enhanced_mag * enhanced_phase, hop_length=hop_length, win_length=win_length, length=len(noisy))


    #cplx map
    # enhanced = stft.istft(out)
    # enhanced = enhanced.squeeze(0).detach().cpu().numpy()  # [1, T] => [T]

    # assert len(noisy) == len(enhanced)

    return noisy, enhanced

