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


def load_noise_file(file_path, sr=32000):
    basename_text = os.path.basename(os.path.splitext(file_path)[0])
    y, _ = librosa.load(file_path, sr=sr)
    return {
        "name": basename_text,
        "y": y
    }


clean_path = r'D:\database\test\clean'
noise_path = r"D:\database\wind_noise\test"  # 噪声文件存放路径
out_path = r'D:\database\test\clean_with_noise'

# 过滤只保留音频文件
files = [f for f in os.listdir(clean_path) if f.endswith('.wav')]
l = len(files)
print(f"Number of clean audio files: {l}")

# 加载噪声数据集
n_jobs = -1
# 从噪声目录中列出所有wav文件
noise_f_paths = [os.path.join(noise_path, f) for f in os.listdir(noise_path) if f.endswith('.wav')]
all_noise_data = Parallel(n_jobs=n_jobs)(
    delayed(load_noise_file)(f_path, sr=32000) for f_path in tqdm(noise_f_paths, desc=f"Loading noise files"))

for i in range(l):
    # 原始语音
    path = os.path.join(clean_path, files[i])

    try:
        a, a_sr = librosa.load(path, sr=32000)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        continue  # 如果加载失败，跳过当前文件

    # 随机选择一种噪声
    noise_data = random.choice(all_noise_data)
    noise_name = noise_data["name"]
    noise_y = noise_data["y"]

    # 随机切片噪声以匹配语音长度
    if len(noise_y) < len(a):
        # 如果噪声长度小于语音长度，循环噪声
        pad_factor = len(a) // len(noise_y) + 1
        noise_y = np.tile(noise_y, pad_factor)[:len(a)]

    s = random.randint(0, len(noise_y) - len(a))  # 随机起始点
    noise_y = noise_y[s:s + len(a)]  # 截取与语音长度相同的噪声片段

    snr_list = ["-5", "0", "5", "10"]
    # 随机选择信噪比
    snr = random.choice(snr_list)

    # 合成
    noisy_y = add_noise_for_waveform(a, noise_y, float(snr))

    noisy_path = os.path.join(out_path, files[i].split('.')[0] + '_' + noise_name[:8] + '_sn' + snr + '.wav')
    sf.write(noisy_path, noisy_y, 32000)

print("Processing complete!")