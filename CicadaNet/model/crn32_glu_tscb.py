
import torch
import torch.nn as nn
from model.conformer import ConformerBlock


class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        self.freq_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
    def forward(self, x_in):
        b, c, f, t = x_in.size()
        x_t = x_in.permute(0, 2, 3, 1).contiguous().view(b*f, t, c)
        # print(f"x_t.shape: {x_t.shape} ")
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        # print(f"x_f.shape: {x_f.shape} ")
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 2, 1)
        return x_f

#two time-conformer
# class TSCB(nn.Module):
#     def __init__(self, num_channel=64):
#         super(TSCB, self).__init__()
#         self.time_conformer1 = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
#                                              conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
#         self.time_conformer2 = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
#                                              conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
#
#     def forward(self, x_in):
#         b, c, f, t = x_in.size()
#         x_t = x_in.permute(0, 2, 3, 1).contiguous().view(b*f, t, c)
#         # print(f"x_t.shape: {x_t.shape} ")
#         x_t1 = self.time_conformer1(x_t) + x_t
#         x_t2 = self.time_conformer2(x_t1) + x_t1
#         out = x_t2.view(b, f, t, c).permute(0, 3, 1, 2)
#         # print(out.shape)
#
#         return out

# two freq conformer
# class TSCB(nn.Module):
#     def __init__(self, num_channel=64):
#         super(TSCB, self).__init__()
#         self.freq_conformer1 = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
#                                              conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
#         self.freq_conformer2 = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
#                                              conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
#     def forward(self, x_in):
#         b, c, f, t = x_in.size()
#         x_f = x_in.permute(0, 3, 2, 1).contiguous().view(b*t, f, c)
#         x_f1 = self.freq_conformer1(x_f) + x_f
#         x_f2 = self.freq_conformer2(x_f1) + x_f1
#         # print(f"x_f.shape: {x_f.shape} ")
#         out = x_f2.view(b, t, f, c).permute(0, 3, 2, 1)
#
#         return out

# freq-time-conformer
# class TSCB(nn.Module):
#     def __init__(self, num_channel=64):
#         super(TSCB, self).__init__()
#         self.time_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
#                                              conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
#         self.freq_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
#                                              conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
#
#     def forward(self, x_in):
#         b, c, f, t = x_in.size()
#         x_f = x_in.permute(0, 3, 2, 1).contiguous().view(b*t, f, c)
#         # print(f"x_f.shape: {x_f.shape} ")
#         #
#         x_f = self.freq_conformer(x_f) + x_f
#         x_t = x_f.view(b, t, f, c).permute(0, 2, 1, 3).contiguous().view(b*f, t, c)
#         #
#         # print(f"x_t.shape: {x_t.shape} ")
#         x_t = self.time_conformer(x_t) + x_t
#         out = x_t.view(b, f, t, c).permute(0, 3, 1, 2)
#         # print(f"out.shape: {out.shape} ")
#
#         return out




class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        # print(f"x.shape: {x.shape} ")
        x1 = self.conv1(x)
        x1 = x1[:, :, :, :-1]  # chomp size
        # x1 = self.norm(x1)
        # x1 = self.activation(x1)
        # print(f"x1.shape: {x1.shape} ")

        x2 = self.conv2(x)
        x2 = x2[:, :, :, :-1]  # chomp size
        x2 = self.sigmoid(x2) #GLU-Conv

        # x2 = self.norm(x2)
        # x2 = self.activation(x2)
        # print(f"x2.shape: {x2.shape} ")
        x = x1*x2
        x = self.norm(x)
        x = self.activation(x)

        # print(f"x1*x2.shape: {x.shape} ")
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.sigmoid = nn.Sigmoid()
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        # print(f"x.shape: {x.shape} ")
        x1 = self.conv1(x)
        x1 = x1[:, :, :, :-1]  # chomp size
        # x1 = self.norm(x1)
        # x1 = self.activation(x1)
        # print(f"x1.shape: {x1.shape} ")

        x2 = self.conv2(x)
        x2 = x2[:, :, :, :-1]  # chomp size
        x2 = self.sigmoid(x2)  # GLU-Conv

        # x2 = self.norm(x2)
        # x2 = self.activation(x2)
        # print(f"x2.shape: {x2.shape} ")
        x = x1*x2
        x = self.norm(x)
        x = self.activation(x)
        # print(f"x1*x2.shape: {x.shape} ")
        return x

class CRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, num_channel=256):
        super(CRN, self).__init__()
        # Encoder
        self.conv_block_1 = CausalConvBlock(1, 8)
        self.conv_block_2 = CausalConvBlock(8, 16)
        self.conv_block_3 = CausalConvBlock(16, 32)
        self.conv_block_4 = CausalConvBlock(32, 64)
        self.conv_block_5 = CausalConvBlock(64, 128)
        self.conv_block_6 = CausalConvBlock(128, 256)

        # LSTM
        # self.lstm_layer = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)
        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        # self.TSCB_3 = TSCB(num_channel=num_channel)
        # self.TSCB_4 = TSCB(num_channel=num_channel)


        self.tran_conv_block_1 = CausalTransConvBlock(256 + 256, 128)
        self.tran_conv_block_2 = CausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = CausalTransConvBlock(64 + 64, 32)
        self.tran_conv_block_4 = CausalTransConvBlock(32 + 32, 16)
        self.tran_conv_block_5 = CausalTransConvBlock(16 + 16, 8, output_padding=(1, 0))
        self.tran_conv_block_6 = CausalTransConvBlock(8 + 8, 1, is_last=True)
        # self.tran_conv_block_6 = CausalTransConvBlock(8 + 8, 1)

    def forward(self, x):
        # self.lstm_layer.flatten_parameters()

        e_1 = self.conv_block_1(x)
        e_2 = self.conv_block_2(e_1)
        e_3 = self.conv_block_3(e_2)
        e_4 = self.conv_block_4(e_3)
        e_5 = self.conv_block_5(e_4)  # [2, 256, 4, 200] [b, c, f, t]
        e_6 = self.conv_block_6(e_5)  # [2, 256, 4, 200] [b, c, f, t]
        # e_6_1 = e_6.permute(0, 1, 3, 2)  # [b, c, t, f]
        # print(f"e_6: {e_6.shape}")

        out_1 = self.TSCB_1(e_6)
        # print(f"out_1: {out_1.shape}")
        out_2 = self.TSCB_1(out_1)    #[b, c, t, f]
        # out_2 = out_2.permute(0, 1, 3, 2)  # [b, c, f, t]
        # print(f"out_2: {out_2.shape}")

        d_1 = self.tran_conv_block_1(torch.cat((out_2, e_6), 1))
        d_2 = self.tran_conv_block_2(torch.cat((d_1, e_5), 1))
        d_3 = self.tran_conv_block_3(torch.cat((d_2, e_4), 1))
        d_4 = self.tran_conv_block_4(torch.cat((d_3, e_3), 1))
        d_5 = self.tran_conv_block_5(torch.cat((d_4, e_2), 1))
        d_6 = self.tran_conv_block_6(torch.cat((d_5, e_1), 1))
        # print(f"d_6: {d_6.shape}")

        return d_6


if __name__ == '__main__':
    layer = CRN()
    a = torch.rand(2, 1, 321, 200)
    print(layer(a).shape)
