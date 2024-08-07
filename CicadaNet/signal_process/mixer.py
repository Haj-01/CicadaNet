import numpy as np
import soundfile as sf
import librosa
import random
import os
from joblib import Parallel, delayed
from tqdm import tqdm

def add_noise_for_waveform(s, n, db):
    alpha = np.sqrt(
        np.sum(s ** 2) / (np.sum(n ** 2) * 10 ** (db / 10))
    )
    mix = s + alpha * n
    return mix


clean_path = r'G:\JNT\dataset\lifan\鸟声数据'
noise_path = r"G:\JNT\dataset\lifan\噪声数据"
clean_seg_path = r'G:\JNT\dataset\lifan\dataset\train\clean\clean5'
noisy_path = r'G:\JNT\dataset\lifan\dataset\train\noisy\noisy5'

clean_file_list = []
clean_files = os.listdir(clean_path)
for file in clean_files:
    clean_file_list.append(os.path.join(clean_path, file))

noise_file_list = []

noise_files = os.listdir(noise_path)
for file in noise_files:
    noise_file_list.append(os.path.join(noise_path, file))

np.random.shuffle(clean_file_list)
np.random.shuffle(noise_file_list)

cleanlen = len(clean_file_list)
noiselen = len(noise_file_list)

fs = 32000
length_per_sample = 4
points_per_sample = length_per_sample * fs

for i in range(0, cleanlen):

    clean_f = clean_file_list[i]
    idx_noise = random.randint(0, (noiselen - 1))
    noise_f = noise_file_list[idx_noise]

    # 随机截取数据
    Begin_S = int(np.random.uniform(0, (10 - length_per_sample) * fs))
    clean_s = sf.read(clean_f, dtype='float32', start=Begin_S, stop=Begin_S + points_per_sample)[0]
    # clean_s = sf.read(clean_f, dtype='float32')[0]
    Begin_N = int(np.random.uniform(0, (10 - length_per_sample) * fs))
    noise_s = sf.read(noise_f, dtype='float32', start=Begin_N, stop=Begin_N + points_per_sample)[0]

    #合成
    # SNR = np.random.uniform(-5, 10)

    snr_list = ["-5", "-2.5", "0", "2.5", "5", "7.5", "10"]
    #随机挑选信噪比
    SNR = random.choice(snr_list)

    noisy_y = add_noise_for_waveform(clean_s, noise_s, float(SNR))

    basefilename = os.path.basename(clean_f)
    basename, ext = os.path.splitext(basefilename)
    new_basefilename = basename + '.wav'
    out_clean_path = os.path.join(clean_seg_path, new_basefilename)
    out_path = os.path.join(noisy_path, new_basefilename)

    sf.write(out_clean_path, clean_s, fs)
    sf.write(os.path.join(out_path.split('.')[0] + '.wav'), noisy_y, fs)






