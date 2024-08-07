import os

import librosa
from torch.utils import data
import torch
import torchaudio
import random
import numpy as np
from util.stft import STFT

class Dataset(data.Dataset):
    def __init__(
            self,
            dataset_list,
            limit,
            offset,
            sr,
            n_fft,
            hop_length,
            win_length,
            train
    ):
        """
        dataset_list(*.txt):
            <noisy_path> <clean_path>\n
        e.g:
            noisy_1.wav clean_1.wav
            noisy_2.wav clean_2.wav
            ...
            noisy_n.wav clean_n.wav
        """
        super(Dataset, self).__init__()
        self.sr = sr
        self.train = train

        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset_list)), "r")]
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.dataset_list = dataset_list
        self.length = len(self.dataset_list)
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.cut_len = 4*16000
        # self.stft = STFT(win_size=320, hop_size=160)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_path, clean_path = self.dataset_list[item].split(" ")
        name = os.path.splitext(os.path.basename(noisy_path))[0]
        frame_mask_list = []
        # noisy_ds, _ = librosa.load(os.path.abspath(os.path.expanduser(noisy_path)), sr=self.sr)
        # clean_ds, _ = librosa.load(os.path.abspath(os.path.expanduser(clean_path)), sr=self.sr)
        # print(f"noisy_ds {noisy_ds.shape}")

        clean_ds, _ = torchaudio.load(clean_path)
        noisy_ds, _ = torchaudio.load(noisy_path)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        minlen = min(len(clean_ds), len(noisy_ds))
        clean_ds = clean_ds[:minlen]
        noisy_ds = noisy_ds[:minlen]
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
            # clean_ds = np.concatenate(clean_ds_final, axis=-1)
            # noisy_ds = np.concatenate(noisy_ds_final, axis=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
        # print(f"noisy_ds {noisy_ds.shape}")
        # noisy_ds = noisy_ds.numpy()
        # clean_ds = clean_ds.numpy()
        # print(f"noisy_ds {noisy_ds.shape}")
        frame_num = (len(noisy_ds) - self.win_length + self.n_fft) // self.hop_length + 1 - 4  ##401
        if self.train:
            # noisy_ds = noisy_ds.numpy()
            # clean_ds = clean_ds.numpy()
            # real_mix, imag_mix = self.stft.stft(noisy_ds)
            # noisy = torch.stack([real_mix, imag_mix], dim=1)
            # print(f"noisy: {noisy.shape} ")
            #
            # real_sph, imag_sph = self.stft.stft(clean_ds)
            # clean = torch.stack([real_sph, imag_sph], dim=1)

            # noisy_ri = librosa.stft(noisy_ds, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft)
            # # noisy = torch.stack([real_noisy, imag_noisy], dim=1)
            # print(f"noisy_ri: {noisy_ri.shape} ")
            # real_clean, imag_clean = librosa.stft(clean_ds, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft)
            # frame_num = (len(noisy_ds) - self.win_length + self.n_fft) // self.hop_length + 1 - 4    ##401
            # frame_mask_list.append(frame_num)
            # print(f"frame_num:{frame_num}")

            # noisy_mag, _ = librosa.magphase(librosa.stft(noisy_ds, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft))
            # clean_mag, _ = librosa.magphase(librosa.stft(clean_ds, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft))
            # return noisy_mag, clean_mag, noisy_mag.shape[-1], name
            # print(f"noisy_mag: {noisy_mag.shape[-1]}")   #401
            return noisy_ds, clean_ds, frame_num
        else:
            return noisy_ds, clean_ds, frame_num, name


