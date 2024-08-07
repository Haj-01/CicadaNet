import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from uformer.loss import sisnr, calloss, calloss_cplxmse, calloss_magmse, calloss_timemae

def mse_loss():
    return torch.nn.MSELoss()


def mse_loss_for_variable_length_data():
    def loss_function(output, src, output_cplx, src_cplx):
        # loss1 = calloss(output, src)
        # print(f"loss1:{loss1}")
        # minlen = min(len(output), len(src))
        # output = output[:, minlen]
        # src = src[:, minlen]
        loss2 = calloss_timemae(output, src)
        # print(f"loss2:{loss2}")
        loss3 = calloss_cplxmse(output_cplx, src_cplx)
        # print(f"loss3:{loss3}")
        loss4 = calloss_magmse(output_cplx, src_cplx)
        # print(f"loss4:{loss4}")
        # loss = float(5)*loss1 + loss3 + loss4
        loss = 0.01*loss2 + loss3 + loss4
        # print(f" loss2:{loss2} loss3:{loss3} loss4:{loss4} loss:{loss}")
        # print(f" loss2:{0.01*loss2} loss3:{loss3} loss4:{loss4} loss:{loss}")
        # print(loss.type())
        # print(f"loss{loss.shape}")

        return loss

    return loss_function

#复数+幅度 损失
def com_mag_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    mag_esti, mag_label = torch.norm(esti, dim=1), torch.norm(label, dim=1)
    loss1 = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
    loss2 = (((mag_esti - mag_label) * mask_for_loss) ** 2).sum() / mask_for_loss.sum()
    # return 0.5 * (loss1 + loss2)
    return loss1, loss2


def LossFunction():
    def com_mag_mse_loss(esti, label, frame_list):
        mask_for_loss = []
        utt_num = esti.size()[0]
        with torch.no_grad():
            for i in range(utt_num):
                tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
                mask_for_loss.append(tmp_mask)
            mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
            com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
        mag_esti, mag_label = torch.norm(esti, dim=1), torch.norm(label, dim=1)
        loss1 = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
        loss2 = (((mag_esti - mag_label) * mask_for_loss) ** 2).sum() / mask_for_loss.sum()
        return 0.5 * (loss1 + loss2)

    return com_mag_mse_loss


# def mse_loss_for_variable_length_data():
#     def loss_function(target, ipt, n_frames_list, device):
#         """
#         Calculate the MSE loss for variable length dataset.
#
#         ipt: [B, F, T] clean
#         target: [B, F, T] enhanced
#         """
#         if target.shape[0] == 1:
#             return torch.nn.functional.mse_loss(target, ipt)
#
#         E = 1e-8
#         with torch.no_grad():
#             masks = []
#             for n_frames in n_frames_list:
#                 masks.append(torch.ones(n_frames, target.size(1), dtype=torch.float32))  # the shape is (T_real, F)
#
#             binary_mask = pad_sequence(masks, batch_first=True).to(device).permute(0, 2, 1)  # ([T1, F], [T2, F]) => [B, T, F] => [B, F, T]
#             # binary_mask = pad_sequence(masks, batch_first=True).to(device)  # ([T1, F], [T2, F]) => [B, T, F] => [B, F, T]
#
#         # print(f"ipt: {ipt.shape}")
#         # print(f"binary_mask: {binary_mask.shape}")  # torch.Size([4, 1, 401])
#         masked_ipt = ipt * binary_mask  # [B, F, T] torch.Size([4, 4, 161, 401])
#         # print(f"masked_ipt: {masked_ipt.shape}")
#         masked_target = target * binary_mask
#         loss = ((masked_ipt - masked_target) ** 2).sum() / (binary_mask.sum() + E)  # 不算 pad 部分的贡献，仅计算有效值
#         # print(f"loss: {loss1}")
#         # loss2 =
#         # return ((masked_ipt - masked_target) ** 2).sum() / (binary_mask.sum() + E)  # 不算 pad 部分的贡献，仅计算有效值
#         return loss
#
#     return loss_function

#原文loss

# class LossFunction(object):
#     def __call__(self, est, lbl, loss_mask, n_frames):
#         est_t = est * loss_mask
#         lbl_t = lbl * loss_mask
#
#         n_feats = est.shape[-1]
#
#         loss = torch.sum((est_t - lbl_t) ** 2) / float(sum(n_frames) * n_feats)
#
#         return loss