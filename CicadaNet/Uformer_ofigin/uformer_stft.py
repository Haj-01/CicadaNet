#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import os
import sys

import torch_complex
from torch_complex import ComplexTensor
import warnings
from time import time

EPSILON = torch.finfo(torch.float32).eps
sys.path.append(os.path.dirname(sys.path[0]) + '/model')

# from Uformer_ofigin.trans import STFT, iSTFT, MelTransform, inv_MelTransform
from util.stft import STFT
from Uformer_ofigin.conv2d_cplx import ComplexConv2d_Encoder, ComplexConv2d_Decoder
from Uformer_ofigin.conv2d_real import RealConv2d_Encoder, RealConv2d_Decoder
from Uformer_ofigin.dilated_dualpath_conformer import Dilated_Dualpath_Conformer
from Uformer_ofigin.fusion import fusion as fusion
from Uformer_ofigin.show import show_model, show_params

def tanhextern(input):
    out = 10 * (1 - torch.exp(-0.1 * input)) / (1 + torch.exp(-0.1 * input))

class Complex_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(Complex_Decoder, self).__init__()
        self.conv1 = ComplexConv2d_Decoder(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            output_padding=output_padding,
                            dilation=(1, 1),
                            groups=1
                        )

        self.norm = nn.BatchNorm3d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = self.prelu(out)

        return out

class mag_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(mag_Decoder, self).__init__()
        self.conv1 = RealConv2d_Decoder(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            output_padding=output_padding,
                            dilation=(1, 1),
                            groups=1
                        )

        self.norm = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = self.prelu(out)

        return out


class Uformer(nn.Module):

    def __init__(self,
                 win_len=640,
                 win_inc=320,
                 fft_len=512,
                 win_type='hanning',
                 fid=None):
        super(Uformer, self).__init__()
        input_dim = win_len
        output_dim = win_len
        self.kernel_num = [1, 8, 16, 32, 64, 128, 128]
        self.kernel_num_real = [1, 8, 16, 32, 64, 128]

        self.encoder = nn.ModuleList()
        self.decoder1 = nn.ModuleList()
        self.decoder2 = nn.ModuleList()
        self.encoder_real = nn.ModuleList()
        self.decoder_real = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d_Encoder(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(5, 2),
                        stride=(2, 1),
                        padding=(2, 1),
                        dilation=(1, 1),
                        groups=1
                    ),
                    nn.BatchNorm3d(self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )

        for idx in range(len(self.kernel_num) - 1):
            self.encoder_real.append(
                nn.Sequential(
                    RealConv2d_Encoder(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(5, 2),
                        stride=(2, 1),
                        padding=(2, 1),
                        dilation=(1, 1),
                        groups=1
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )

        self.conformer = Dilated_Dualpath_Conformer()

        #win_len=640, decoder1_1: output_padding=(0, 0)
        self.decoder1_1 = Complex_Decoder(256, 128, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0),
                                        output_padding=(0, 0))
        self.decoder1_2 = Complex_Decoder(256, 64, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0),
                                        output_padding=(0, 0))
        self.decoder1_3 = Complex_Decoder(128, 32, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0),
                                        output_padding=(0, 0))
        self.decoder1_4 = Complex_Decoder(64, 16, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0),
                                        output_padding=(0, 0))
        self.decoder1_5 = Complex_Decoder(32, 8, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0),
                                        output_padding=(0, 0))
        self.decoder1_6 = Complex_Decoder(16, 1, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0),
                                        output_padding=(0, 0))

        self.decoder2_1 = mag_Decoder(256, 128, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0),
                                        output_padding=(0, 0))
        self.decoder2_2 = mag_Decoder(256, 64, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0),
                                        output_padding=(0, 0))
        self.decoder2_3 = mag_Decoder(128, 32, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0),
                                        output_padding=(0, 0))
        self.decoder2_4 = mag_Decoder(64, 16, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0),
                                        output_padding=(0, 0))
        self.decoder2_5 = mag_Decoder(32, 8, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0),
                                        output_padding=(0, 0))
        self.decoder2_6 = mag_Decoder(16, 1, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0),
                                        output_padding=(0, 0))

        self.stft = STFT(win_size=win_len, hop_size=win_inc)
        # self.istft = iSTFT(frame_len=win_len, frame_hop=win_inc)

    def flatten_parameters(self):
        self.enhance.flatten_parameters()

    def forward(self, inputs):
        warnings.filterwarnings('ignore')

        # print(f"inputs: {inputs.shape}")
        inputs_real, inputs_imag = self.stft.stft(inputs)
        inputs_real = inputs_real.unsqueeze(1).permute(0, 1, 3, 2)   ##torch.Size([8, 1, 161, 401])
        inputs_imag = inputs_imag.unsqueeze(1).permute(0, 1, 3, 2)
        # src_real, src_imag = self.stft.stft(src)   #torch.Size([8, 401, 161])
        # src_cplx = torch.stack([src_real, src_imag], 1)  #torch.Size([8, 2, 401, 161])
        #
        # src = self.stft.istft(src_cplx)
        # src_cplx = src_cplx.permute(0, 1, 3, 2)  #torch.Size([8, 2, 161, 401])

        # print(f"src: {src.shape}")
        # src_mag, src_pha = torch.sqrt(torch.clamp(src_real ** 2 + src_imag ** 2, EPSILON)), torch.atan2(
        #     src_imag + EPSILON, src_real)
        #
        # src_mag = src_mag ** 0.5
        # src_real, src_imag = src_mag * torch.cos(src_pha), src_mag * torch.sin(src_pha)
        # src_cplx = torch.stack([src_real, src_imag], 1)

        mag, phase = torch.sqrt(torch.clamp(inputs_real ** 2 + inputs_imag ** 2, EPSILON)), torch.atan2(
            inputs_imag + EPSILON, inputs_real)
        # mag = mag ** 0.5    #torch.Size([8, 1, 161, 401])
        # print(f"mag: {mag.shape}")
        mag_input = []

        mag_input.append(mag)

        # inputs_real, inputs_imag = mag * torch.cos(phase), mag * torch.sin(phase)

        out = torch.stack([inputs_real, inputs_imag], -1)  # B C F T 2  torch.Size([8, 1, 161, 401, 2])

        encoder_out = []
        mag_out = []

        for idx in range(len(self.encoder)):
            out = self.encoder[idx](out)
            # print(f"{idx} encoder_out: {out.shape}")
            mag = self.encoder_real[idx](mag)
            out, mag = fusion(out, mag)   #torch.Size([8, 128, 3, 401]), [8, 128, 3, 401, 2]
            mag_out.append(mag)
            encoder_out.append(out)

        out, mag = self.conformer(out, mag)  ##torch.Size([8, 128, 3, 401]), [8, 128, 3, 401, 2]

        de1 = self.decoder1_1(torch.cat([encoder_out[5], out], 1))
        de_mag1 = self.decoder2_1(torch.cat([mag_out[5], mag], 1))
        de1, de_mag1 = fusion(de1, de_mag1)

        de2 = self.decoder1_2(torch.cat([encoder_out[4], de1], 1))
        de_mag2 = self.decoder2_2(torch.cat([mag_out[4], de_mag1], 1))
        de2, de_mag2 = fusion(de2, de_mag2)

        de3 = self.decoder1_3(torch.cat([encoder_out[3], de2], 1))
        de_mag3 = self.decoder2_3(torch.cat([mag_out[3], de_mag2], 1))
        de3, de_mag3 = fusion(de3, de_mag3)

        de4 = self.decoder1_4(torch.cat([encoder_out[2], de3], 1))
        de_mag4 = self.decoder2_4(torch.cat([mag_out[2], de_mag3], 1))
        de4, de_mag4 = fusion(de4, de_mag4)

        de5 = self.decoder1_5(torch.cat([encoder_out[1], de4], 1))
        de_mag5 = self.decoder2_5(torch.cat([mag_out[1], de_mag4], 1))
        de5, de_mag5 = fusion(de5, de_mag5)

        de6 = self.decoder1_6(torch.cat([encoder_out[0], de5], 1))
        de_mag6 = self.decoder2_6(torch.cat([mag_out[0], de_mag5], 1))
        out, mag = fusion(de6, de_mag6)

        mag = torch.sigmoid(mag)   #公式15
        mag = mag[:, 0] * mag_input[0][:, 0]  #公式17

        mask_real = out[..., 0]
        mask_imag = out[..., 1]

        mask_mags = torch.sqrt(torch.clamp(mask_real ** 2 + mask_imag ** 2, EPSILON))   #公式12_1
        real_phase = mask_real / (mask_mags + EPSILON)
        imag_phase = mask_imag / (mask_mags + EPSILON)
        mask_mags = torch.tanh(mask_mags + EPSILON)   ##公式12_2
        mask_phase = torch.atan2(imag_phase + EPSILON, real_phase)   #公式13

        est_mags = mask_mags[:, 0] * mag_input[0][:, 0]   #公式15

        est_phase = phase[:, 0] + mask_phase[:, 0]     #公式16

        mag_compress, pha_compress = est_mags, est_phase
        mag_compress = (mag_compress + mag) * 0.5    #公式18

        real, imag = mag_compress * torch.cos(pha_compress), mag_compress * torch.sin(pha_compress)   #公式19，20
        output_real = []
        output_imag = []
        output_real.append(real)
        output_imag.append(imag)
        output_real = torch.stack(output_real, 1)
        output_imag = torch.stack(output_imag, 1)
        output_real = output_real.squeeze(1)  # N x C x F x T
        output_imag = output_imag.squeeze(1)
        output_cplx = torch.stack([output_real, output_imag], 1)  # N x 2 x F x T  torch.Size([8, 2, 161, 401])

        output = []
        output_cplx1 = output_cplx.permute(0, 1, 3, 2)   #torch.Size([8, 2, 401, 161])
        spk1 = self.stft.istft(output_cplx1)
        output.append(spk1)
        output = torch.stack(output, 1)
        output = output.squeeze(1)
        # print(f"output: {output.shape}")

        return output, output_cplx


if __name__ == '__main__':
    torch.manual_seed(10)
    torch.set_num_threads(4)

    import soundfile
    import numpy as np

    net = Uformer()
    stft = STFT(win_size=320, hop_size=160)
    inputs = torch.randn([8, 64000])
    clean_path = r"E:\JNT\vctk\test\p232_001.wav"
    enhanced_dir = r"E:\JNT\vctk\test\p232_001_output_cplx.wav"
    clean, _ = torchaudio.load(clean_path)
    print(clean.shape)
    output, output_cplx = net(inputs)
    # print('output_cplx: ', output_cplx.shape)
    # print('src_cplx: ', src_cplx.shape)
    # spk1 = stft.istft(output_cplx)
    # enhanced_wav = spk1.detach().cpu().numpy().reshape(-1)
    # print('enhanced_wav: ', enhanced_wav.shape)
    # soundfile.write(enhanced_dir, enhanced_wav, 16000)
    print('output: ', output.shape)
    print('output_cplx: ', output_cplx.shape)



