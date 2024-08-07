# import pysepm
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import librosa
import pandas as pd
import xlwt
# import pysepm
EPSILON = 1e-8


clean_wavs = r"G:\JNT\test_dataset\clean"# 这个路径，大家根据自己的需求进行修改
denoised_wavs = r"G:\JNT\test_dataset\enhanced\cicadanet_mag_map_magloss_11tscb\sn0"# 同理进行修改

excelName = r"D:\1testing\1normalize\noisy_sn0.xls"

def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.wav':
                L.append(os.path.join(root, file))
    return L

def SI_SDR(reference, estimation):
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



def l2_norm(s1, s2):
    # norm = np.sqrt(np.sum(s1*s2, 1, keepdims=True))
    norm = np.linalg.norm(s1*s2, 1, keepdims=True)

    # norm = np.sum(s1 * s2, -1, keepdims=True)
    return norm

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
    snr = 10 * np.log10((target_norm) / (noise_norm + eps) + eps)
    return np.mean(snr)


def extract_overlapped_windows(x,nperseg,noverlap,window=None):
    # source: https://github.com/scipy/scipy/blob/v1.2.1/scipy/signal/spectral.py
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    if window is not None:
        result = window * result
    return result

def SNRseg(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):
    eps = np.finfo(np.float64).eps

    winlength = round(frameLen * fs)  # window length in samples
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))  # window skip in samples
    MIN_SNR = -10  # minimum SNR in dB
    MAX_SNR = 35  # maximum SNR in dB

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    clean_speech_framed = extract_overlapped_windows(clean_speech, winlength, winlength - skiprate, hannWin)
    processed_speech_framed = extract_overlapped_windows(processed_speech, winlength, winlength - skiprate, hannWin)

    signal_energy = np.power(clean_speech_framed, 2).sum(-1)
    noise_energy = np.power(clean_speech_framed - processed_speech_framed, 2).sum(-1)

    segmental_snr = 10 * np.log10(signal_energy / (noise_energy + eps) + eps)
    segmental_snr[segmental_snr < MIN_SNR] = MIN_SNR
    segmental_snr[segmental_snr > MAX_SNR] = MAX_SNR
    segmental_snr = segmental_snr[:-1]  # remove last frame -> not valid
    return np.mean(segmental_snr)


def compute_snr(ref, est):
    ratio = 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - est) ** 2))
    return ratio


cleanfiles = os.listdir(clean_wavs)
denoisefiles = os.listdir(denoised_wavs)
zipped = zip(cleanfiles, denoisefiles)
l = len(cleanfiles)
sisdrs = []
SNRsegs = []
sisnrs = []
snrs = []

f = xlwt.Workbook(encoding='utf-8', style_compression=0)  # 新建一个excel
sheet = f.add_sheet('sheet1')  # 新建一个sheet
i = 0

for (clean_wav, denoised_wav) in tqdm(zipped, 'the progressing ...'):
    # for i in range(0, l):
    # for (clean_wav, denoised_wav) in tqdm(zipped, 'the progressing ...'):
    # Gain speech parameters
    # print(f"clean_wav:{clean_wav} denoised_wav:{denoised_wav}")
    clean_path = os.path.join(clean_wavs, clean_wav)
    denoise_path = os.path.join(denoised_wavs, denoised_wav)
    ref, sr0 = librosa.load(clean_path, sr=32000)
    deg, sr1 = librosa.load(denoise_path, sr=32000)
    # ref, sr0 = sf.read(clean_wav)
    # deg, sr1 = sf.read(denoised_wav)
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]

    snr = compute_snr(ref, deg)
    snrs.append(snr)

    # sisdr = SI_SDR(ref, deg)
    # sisdrs.append(sisdr)

    sisnr = si_snr(deg, ref)
    sisnr = sisnr.astype(np.float64)
    sisnrs.append(sisnr)

    sisdr = SI_SDR(deg, ref)
    sisdrs.append(sisdr)

    SNRseg_score = SNRseg(ref, deg, sr0)
    SNRsegs.append(SNRseg_score)

    # if sisdr>0 and SNRseg_score>-6:
    #     # print(f"clean_wav{clean_wav},denoised_wav{denoised_wav}")
    #     sheet.write(i, 0, clean_wav)  # 参数i,0,s分别代表行，列，写入值
    #     sheet.write(i, 1, sisdr)  # 参数i,0,s分别代表行，列，写入值
    #     sheet.write(i, 2, SNRseg_score)  # 参数i,0,s分别代表行，列，写入值
    #     i = i + 1
    #     f.save(excelName)

    # sheet.write(i, 0, clean_wav)  # 参数i,0,s分别代表行，列，写入值
    # sheet.write(i, 1, snr)  # 参数i,0,s分别代表行，列，写入值
    # sheet.write(i, 2, sisdr)  # 参数i,0,s分别代表行，列，写入值
    # sheet.write(i, 3, sisnr)  # 参数i,0,s分别代表行，列，写入值
    # sheet.write(i, 4, SNRseg_score)  # 参数i,0,s分别代表行，列，写入值
    #
    # i = i + 1
    # f.save(excelName)


print('The average SNR evaluation is :   ', sum(snrs) / len(snrs))  #4.957195496559143
print('The average SI-SDR evaluation is :   ', sum(sisdrs) / len(sisdrs))  #4.957195496559143
print('The average SI-SNR evaluation is :   ', sum(sisnrs) / len(sisnrs))  #4.957195496559143
print('The average SNRseg evaluation is :   ', sum(SNRsegs) / len(SNRsegs))





