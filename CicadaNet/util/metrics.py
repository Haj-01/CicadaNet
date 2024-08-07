import numpy as np
# from pesq import pesq
# from pystoi.stoi import stoi
import librosa
import torch


def si_sdr(reference, estimation):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR
    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) \
                      / reference_energy

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)


# torch sisdr
# def si_sdr(reference, estimation, eps=1e-8):
#     """
#         Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
#         Args:
#             reference: numpy.ndarray, [..., T]
#             estimation: numpy.ndarray, [..., T]
#         Returns:
#             SI-SDR
#         [1] SDR– Half- Baked or Well Done?
#         http://www.merl.com/publications/docs/TR2019-013.pdf
#         >>> np.random.seed(0)
#         >>> reference = np.random.randn(100)
#         >>> si_sdr(reference, reference)
#         inf
#         >>> si_sdr(reference, reference * 2)
#         inf
#         >>> si_sdr(reference, np.flip(reference))
#         -25.127672346460717
#         >>> si_sdr(reference, reference + np.flip(reference))
#         0.481070445785553
#         >>> si_sdr(reference, reference + 0.5)
#         6.3704606032577304
#         >>> si_sdr(reference, reference * 2 + 1)
#         6.3704606032577304
#         >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
#         nan
#         >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
#         array([6.3704606, 6.3704606])
#         :param reference:
#         :param estimation:
#         :param eps:
#         """
#
#     reference_energy = torch.sum(reference ** 2, axis=-1, keepdims=True)
#
#     # This is $\alpha$ after Equation (3) in [1].
#     optimal_scaling = torch.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy + eps
#
#     # This is $e_{\text{target}}$ in Equation (4) in [1].
#     projection = optimal_scaling * reference
#
#     # This is $e_{\text{res}}$ in Equation (4) in [1].
#     noise = estimation - projection
#
#     ratio = torch.sum(projection ** 2, axis=-1) / torch.sum(noise ** 2, axis=-1) + eps
#
#     ratio = torch.mean(ratio)
#     return 10 * torch.log10(ratio + eps)

def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm

def sdr(s1, s2, eps=1e-8):
    sn = l2_norm(s1, s1)
    sn_m_shn = l2_norm(s1 - s2, s1 - s2)
    sdr_loss = 10 * torch.log10(sn**2 / (sn_m_shn**2 + eps) + eps)
    return torch.mean(sdr_loss)

def si_snr(s1, s2, eps=1e-8):
    #s1: estimate
    #s2: reference
    # s1 = remove_dc(s1)
    # s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)


#
# def STOI(ref, est, sr=32000):
#     return stoi(ref, est, sr, extended=False)
#
#
# def PESQ(ref, est, sr=32000):
#     target_sr = 16000
#     ref_rs = librosa.resample(ref, sr, target_sr)
#     est_rs = librosa.resample(est, sr, target_sr)
#     return pesq(target_sr, ref_rs, est_rs, "wb")
