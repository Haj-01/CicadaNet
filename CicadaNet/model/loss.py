import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F


def mse_loss_for_variable_length_data():
    def loss_function(target, ipt, n_frames_list, device):
        """
        Calculate the MSE loss for variable length dataset.

        ipt: [B, F, T]  label
        target: [B, F, T]  est
        """
        if target.shape[0] == 1:
            return torch.nn.functional.mse_loss(target, ipt)

        E = 1e-8
        masks = []
        with torch.no_grad():
            for n_frames in n_frames_list:
                masks.append(torch.ones(n_frames, target.size(1), dtype=torch.float32))  # the shape is (T_real, F)

            binary_mask = pad_sequence(masks, batch_first=True).to(device).permute(0, 2, 1)  # ([T1, F], [T2, F]) => [B, T, F] => [B, F, T]

        masked_ipt = ipt * binary_mask  # [B, F, T]
        masked_target = target * binary_mask
        return ((masked_ipt - masked_target) ** 2).sum() / (binary_mask.sum() + E)  # 不算 pad 部分的贡献，仅计算有效值

    return loss_function


def lossMask(shape, n_frames, device):
    loss_mask = torch.zeros(shape, dtype=torch.float32, device=device)
    for i, seq_len in enumerate(n_frames):
        loss_mask[i,:,0:seq_len,:] = 1.0
    return loss_mask

def mse_loss():
    def loss_function(est, lbl, loss_mask, n_frames):

        est_t = est * loss_mask
        lbl_t = lbl * loss_mask

        n_feats = est.shape[-1]

        loss = torch.sum((est_t - lbl_t) ** 2) / float(sum(n_frames) * n_feats)

        return loss
    return loss_function

def mse_loss_for_mag():
    def mag_mse_loss(esti, label, frame_list, device):
        # B F T
        if esti.shape[0] == 1:
            return torch.nn.functional.mse_loss(esti, label)

        E = 1e-8

        mask_for_loss = []
        utt_num = esti.size()[0]

        with torch.no_grad():
            for i in range(utt_num):
                tmp_mask = torch.ones((frame_list[i], esti.size()[1]), dtype=esti.dtype)   #the shape is (T_real, F)
                mask_for_loss.append(tmp_mask)
            mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(device).permute(0, 2, 1)   # B F T

        loss = (((esti - label) * mask_for_loss) ** 2).sum() / (mask_for_loss.sum() + E)
        return loss

    return mag_mse_loss



# def mse_loss_for_mag():
#     def calloss_magmse(enhanced, clean, n_frames_list, device):
#         # B C F T
#
#         loss_mag = F.mse_loss(clean, enhanced, reduction='mean')
#         return loss_mag
#
#     return calloss_magmse

def spectrum_mse_loss():
    def spectrum_mse(enh_mag, clean_mag, enhanced_cplx, clean_cplx):
        # B C F T
        enh_real, enh_imag = enhanced_cplx[:, 0], enhanced_cplx[:, 1]
        clean_real, clean_imag = clean_cplx[:, 0], clean_cplx[:, 1]

        one1 = torch.ones_like(enh_mag)
        zero1 = torch.zeros_like(enh_mag)
        enh_magmask = torch.where(enh_mag < 5e-4, one1, zero1)
        clean_magmask = torch.where(clean_mag < 5e-4, one1, zero1)
        mymask = -1 * (enh_magmask * clean_magmask) + 1

        # mag_loss = torch.mean(mymask * ((clean_mag - enh_mag) ** 2))
        mag_loss = torch.mean(mymask * ((clean_mag - enh_mag) ** 2 + (clean_real - enh_real) ** 2 + (clean_imag - enh_imag) ** 2))

        # mag_loss = torch.log(mag_loss + 1e-12)

        return mag_loss

    return spectrum_mse






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


def sisnr_loss():
    def si_snr(s1, s2, eps=1e-8):
        # s1: estimate
        # s2: reference
        # s1 = remove_dc(s1)
        # s2 = remove_dc(s2)
        s1_s2_norm = l2_norm(s1, s2)
        s2_s2_norm = l2_norm(s2, s2)
        s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
        e_nosie = s1 - s_target
        target_norm = l2_norm(s_target, s_target)
        noise_norm = l2_norm(e_nosie, e_nosie)
        snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
        return -torch.mean(snr)

    return si_snr
