import librosa
import torch

EPSILON = torch.finfo(torch.float32).eps

def full_band_no_truncation1(model, device, inference_args, noisy):
    """
    extract full_band spectra for inference, without truncation.
    """
    n_fft = inference_args["n_fft"]
    hop_length = inference_args["hop_length"]
    win_length = inference_args["win_length"]

    noisy_mag, noisy_phase = librosa.magphase(librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    noisy_mag = torch.tensor(noisy_mag, device=device)[None, None, :, :]  # [F, T] => [1, 1, F, T]
    enhanced_mag = model(noisy_mag)  # [1, 1, F, T] => [1, 1, F, T]
    enhanced_mag = enhanced_mag.squeeze(0).squeeze(0).detach().cpu().numpy()  # [1, 1, F, T] => [F, T]

    enhanced = librosa.istft(enhanced_mag * noisy_phase, hop_length=hop_length, win_length=win_length, length=len(noisy))

    assert len(noisy) == len(enhanced)

    return noisy, enhanced

def full_band_no_truncation(model, device, inference_args, noisy):
    """
    extract full_band spectra for inference, without truncation.
    """
    n_fft = inference_args["n_fft"]
    hop_length = inference_args["hop_length"]
    win_length = inference_args["win_length"]

    # noisy_mag, noisy_phase = librosa.magphase(librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    noisy1 = torch.tensor(noisy, device=device)[None, None, :]  # [N] => [1, 1, N]
    # print(f"noisy1:{noisy1.shape}")
    # clean1 = torch.tensor(clean, device=device)[None, None, :]  # [F, T] => [1, 1, N]
    output_cplx, mag = model(noisy1)  # [1, 1, F, T] => [1, 1, F, T]
    enhanced_real, enhanced_imag = output_cplx[:, 0], output_cplx[:, 1]
    enhanced_mag, enhanced_pha = torch.sqrt(torch.clamp(enhanced_real ** 2 + enhanced_imag ** 2, EPSILON)), torch.atan2(
        enhanced_imag + EPSILON, enhanced_real)

    enhanced_mag = enhanced_mag.squeeze(0).squeeze(0).detach().cpu().numpy()
    enhanced_pha = enhanced_pha.squeeze(0).squeeze(0).detach().cpu().numpy()
    enhanced = librosa.istft(enhanced_mag * enhanced_pha, hop_length=hop_length, win_length=win_length,
                             length=len(noisy))
    # output = output.squeeze(0).detach().cpu().numpy()  # [1, 1, F, T] => [F, T]
    # src = src.squeeze(0).detach().cpu().numpy()  # [1, 1, F, T] => [F, T]
    # print(f"output:{output.shape}")
    #
    # print(f"noisy:{noisy.shape}")
    # print(f"clean:{clean.shape}")

    # enhanced = librosa.istft(enhanced_mag * noisy_phase, hop_length=hop_length, win_length=win_length, length=len(noisy))

    # assert len(noisy) == len(output)

    return enhanced
