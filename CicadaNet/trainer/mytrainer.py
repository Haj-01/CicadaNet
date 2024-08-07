import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from trainer.base_trainer import BaseTrainer
from util.my_stft import STFT

plt.switch_backend('agg')

# self.device = torch.self.device("cuda")

class Trainer(BaseTrainer):
    def __init__(self, config, resume: bool, model, loss_function, optimizer, train_dataloader, validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.hop_length = self.validation_custom_config["hop_length"]
        self.win_length = self.validation_custom_config["win_length"]
        self.stft = STFT(win_size=self.win_length, hop_size=self.hop_length).to(self.device)

    def _train_epoch(self, epoch):
        loss_total = 0.0
        batch_num = 0
        for noisy, clean, n_frames_list in self.train_dataloader:
            batch_num += 1
            self.optimizer.zero_grad()

            noisy = noisy.float().to(self.device)
            clean = clean.float().to(self.device)
            noisy_real, noisy_imag = self.stft.stft(noisy)  # B x T x F
            noisy_real = noisy_real.permute(0, 2, 1)
            noisy_imag = noisy_imag.permute(0, 2, 1)
            noisy_phase = torch.atan2(noisy_imag, noisy_real)
            # noisy_cplx = torch.stack([noisy_real, noisy_imag], 1)  # N x 2 x F x T

            noisy_mag = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2 + 1e-8)
            noisy_mag = noisy_mag.unsqueeze(1)

            enhanced_mag = self.model(noisy_mag)  # [B, 1, F, T]

            enhanced_mag = torch.squeeze(enhanced_mag, dim=1)
            enhanced_real, enhanced_imag = enhanced_mag * torch.cos(noisy_phase), enhanced_mag * torch.sin(noisy_phase)
            enhanced_cplx = torch.stack([enhanced_real, enhanced_imag], 1)  # N x 2 x F x T

            clean_real, clean_imag = self.stft.stft(clean)
            clean_real = clean_real.permute(0, 2, 1)
            clean_imag = clean_imag.permute(0, 2, 1)
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-8)    # B F T
            clean_phase = torch.atan2(clean_imag, clean_real)
            # clean_mag = clean_mag.unsqueeze(1)

            clean_real, clean_imag = clean_mag * torch.cos(clean_phase), clean_mag * torch.sin(clean_phase)
            clean_cplx = torch.stack([clean_real, clean_imag], 1)  # N x 2 x F x T

            # enhanced_mag = torch.squeeze(enhanced_mag, dim=1)
            # clean_mag = torch.squeeze(clean_mag, dim=1)

            loss = self.loss_function(enhanced_mag, clean_mag, enhanced_cplx, clean_cplx)
            # loss = self.loss_function(enhanced_mag, clean_mag, n_frames_list, self.device)

            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        loss_total /= batch_num
        print(f"第 {epoch} epoch train loss: {loss_total}")
        self.writer.add_scalar(f"Loss/Train", loss_total, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        noisy_list = []
        clean_list = []
        enhanced_list = []

        loss_total = 0.0
        batch_num = 0

        for noisy, clean, n_frames_list in self.validation_dataloader:

            batch_num += 1

            noisy = noisy.float().to(self.device)
            clean = clean.float().to(self.device)

            noisy_real, noisy_imag = self.stft.stft(noisy)  # B x T x F
            noisy_real = noisy_real.permute(0, 2, 1)
            noisy_imag = noisy_imag.permute(0, 2, 1)

            noisy_mag = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2 + 1e-8)
            noisy_mag = noisy_mag.unsqueeze(1)       # [B, 1, F, T]
            noisy_phase = torch.atan2(noisy_imag, noisy_real)

            enhanced_mag = self.model(noisy_mag)  # [B, 1, F, T]

            enhanced_mag = torch.squeeze(enhanced_mag, dim=1)
            enh_real, enh_imag = enhanced_mag * torch.cos(noisy_phase), enhanced_mag * torch.sin(noisy_phase)
            enh_cplx = torch.stack([enh_real, enh_imag], 1)

            clean_real, clean_imag = self.stft.stft(clean)
            clean_real = clean_real.permute(0, 2, 1)
            clean_imag = clean_imag.permute(0, 2, 1)
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-8)   # B F T
            clean_phase = torch.atan2(clean_imag, clean_real)

            clean_real, clean_imag = clean_mag * torch.cos(clean_phase), clean_mag * torch.sin(clean_phase)
            clean_cplx = torch.stack([clean_real, clean_imag], 1)  # N x 2 x F x T

            # clean_mag = clean_mag.unsqueeze(1)
            # enhanced_mag = torch.squeeze(enhanced_mag, dim=1)
            # clean_mag = torch.squeeze(clean_mag, dim=1)

            # loss_total += self.loss_function(enhanced_mag, clean_mag, n_frames_list, self.device).item()
            loss_total += self.loss_function(enhanced_mag, clean_mag, enh_cplx, clean_cplx).item()

            enh_cplx = torch.transpose(enh_cplx, 2, 3)       # B C T F
            enhanced = self.stft.istft(enh_cplx)

            assert len(noisy) == len(clean) == len(enhanced)


            noisy_list.append(noisy)
            clean_list.append(clean)
            enhanced_list.append(enhanced)

        loss_total /= batch_num

        print(f"第 {epoch} epoch validation loss: {loss_total}")
        self.writer.add_scalar(f"Loss/Validation", loss_total, epoch)
        return self.metrics_visualization(noisy_list, clean_list, enhanced_list, epoch)
