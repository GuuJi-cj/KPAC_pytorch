import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.act = nn.LeakyReLU(negative_slope=1e-2)

    def forward(self, x):
        return self.act(self.conv(x))


class BasicDeConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=2, bias=False):
        super(BasicDeConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ks = kernel_size
        self.stride = stride
        self.bias = bias
        self.deconv = nn.ConvTranspose2d(self.in_channel, self.out_channel, kernel_size=self.ks, stride=self.stride,
                                         padding=self.ks //2 - 1, bias=self.bias)
        self.act = nn.LeakyReLU(negative_slope=1e-2)

    def forward(self, x):
        return self.act(self.deconv(x))


class BasicDilatedConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilaton, bias):
        super(BasicDilatedConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation=dilaton, bias=bias)
        self.act = nn.LeakyReLU(negative_slope=1e-2)

    def forward(self, x):
        return self.act(self.conv(x))


class KpacBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, stride, bias=False, hrg=128, wrg=128):
        super(KpacBlock, self).__init__()
        self.ch = in_channel // 2
        self.bias = bias
        self.hrg = hrg
        self.wrg = wrg
        self.nn1 = nn.Sequential(
            BasicDilatedConv(in_channel, self.ch, kernel_size, stride, padding=2, dilaton=1, bias=self.bias)
        )
        self.nn2 = nn.Sequential(
            BasicDilatedConv(in_channel, self.ch, kernel_size, stride, padding=4, dilaton=2, bias=self.bias)
        )
        self.nn3 = nn.Sequential(
            BasicDilatedConv(in_channel, self.ch, kernel_size, stride, padding=6, dilaton=3, bias=self.bias)
        )
        self.nn4 = nn.Sequential(
            BasicDilatedConv(in_channel, self.ch, kernel_size, stride, padding=8, dilaton=4, bias=self.bias)
        )
        self.nn5 = nn.Sequential(
            BasicDilatedConv(in_channel, self.ch, kernel_size, stride, padding=10, dilaton=5, bias=self.bias)
        )

        self.scale_attention = nn.Sequential(
            BasicDilatedConv(in_channel, 32, kernel_size=kernel_size, stride=1, padding=4, dilaton=2, bias=self.bias),
            BasicDilatedConv(32, 32, kernel_size=kernel_size, stride=1, padding=4, dilaton=2, bias=self.bias),
            BasicDilatedConv(32, 16, kernel_size=kernel_size, stride=1, padding=4, dilaton=2, bias=self.bias),
            BasicDilatedConv(16, 16, kernel_size=kernel_size, stride=1, padding=4, dilaton=2, bias=self.bias),
            nn.Conv2d(16, 5, kernel_size=kernel_size, stride=1, padding=2, bias=self.bias),
            nn.Sigmoid()
        )
        self.upsample_layer1 = nn.Upsample(scale_factor=(self.ch, 1), mode='nearest')

        self.shape_attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=(hrg // 4, wrg // 4), stride=(hrg // 4, wrg // 4)),
            BasicConv(in_channel, self.ch / 4, kernel_size=1, stride=1, padding=0, bias=self.bias),
            nn.Conv2d(self.ch / 4, self.ch, kernel_size=1, stride=1, padding=0, bias=self.bias),
            nn.Sigmoid()
        )
        self.upsample_layer2 = nn.Upsample(scale_factor=(hrg//4, wrg//4), mode='nearest')

        self.conv_out = BasicConv(self.ch * 5, self.ch * 2, kernel_size=3, stride=1, padding=1,
                                  bias=self.bias)

    def forward(self, x):
        nn1_out = self.nn1(x)
        nn2_out = self.nn2(x)
        nn3_out = self.nn3(x)
        nn4_out = self.nn4(x)
        nn5_out = self.nn5(x)

        nn_concat = torch.concat([nn1_out, nn2_out, nn3_out, nn4_out, nn5_out], dim=1)  # (N, 5*ch, H/4, W/4)

        scale_att_map = self.scale_attention(x).permute(0, 2, 1, 3)     # (N, H/4, 5, W/4)
        scale_att_map = self.upsample_layer1(scale_att_map).permute(0, 2, 1, 3)         # (N, 5C, H/4, W/4)
        nn = scale_att_map * nn_concat  # (N, 5C, H/4, W/4)

        shape_att_map = self.shape_attention(x)     # (N, C, 4, 4)
        shape_att_map = shape_att_map.repeat(1, 5, 1, 1)        # (N, 5C, 4, 4)
        shape_att_map = self.upsample_layer(shape_att_map)  # (N, 5C, H/4, W/4)

        nn = shape_att_map * nn

        nn = self.conv_out(nn)

        return nn


class KPAC(nn.Module):
    def __init__(self, ks=5, num_kpac=2, ch=48, hrg=128, wrg=128, bias=False):
        super(KPAC, self).__init__()
        self.ks = ks
        self.num_kpac = num_kpac
        self.ch = ch
        self.bias = bias
        self.conv_in = nn.Sequential(
            BasicConv(3, self.ch, kernel_size=5, stride=1, padding=2, bias=self.bias),
            BasicConv(self.ch, self.ch, kernel_size=3, stride=1, padding=1, bias=self.bias)
        )
        self.encoder_1 = nn.Sequential(
            BasicConv(self.ch, self.ch, kernel_size=3, stride=2, padding=1, bias=self.bias),
            BasicConv(self.ch, self.ch, kernel_size=3, stride=1, padding=1, bias=self.bias)
        )
        self.encoder_2 = nn.Sequential(
            BasicConv(self.ch, self.ch * 2, kernel_size=3, stride=2, padding=1, bias=self.bias),
            BasicConv(self.ch * 2, self.ch * 2, kernel_size=3, stride=1, padding=1, bias=self.bias)
        )

        self.kpac_blocks = []
        for i in range(self.num_kpac):
            self.kpac_blocks.append(KpacBlock(self.ch * 2, kernel_size=self.ks, stride=1, bias=self.bias,
                                              hrg=hrg, wrg=wrg))

        self.conv_out = nn.Sequential(
            BasicConv(self.ch * 6, self.ch * 2, kernel_size=3, stride=1, padding=1, bias=self.bias),
            BasicConv(self.ch * 2, self.ch * 2, kernel_size=3, stride=1, padding=1, bias=self.bias)
        )

        self.decoder_1 = BasicDeConv(self.ch * 2, self.ch, kernel_size=4, stride=2, bias=self.bias)
        self.decoder_conv_1 = BasicConv(self.ch * 2, self.ch, kernel_size=3, stride=1, padding=1, bias=self.bias)

        self.decoder_2 = BasicDeConv(self.ch, self.ch, kernel_size=4, stride=2, bias=self.bias)
        self.decoder_conv_2 = nn.Conv2d(self.ch * 2, 3, kernel_size=5, stride=1, padding=2, bias=self.bias)

    def forward(self, x):
        # Encoder
        inp = x
        x = self.conv_in(x)
        f1 = x
        x = self.encoder_1(x)
        f2 = x
        x = self.encoder_2(x)
        stack = x

        # KPAC Block
        for i in range(self.num_kpac):
            nn = self.kpac_blocks[i](x)     # (N, 2C, H/4, W/4)
            stack = torch.concat([nn, stack], dim=1)    # (N, 6C, H/4, W/4)
            x = nn

        # Decoder
        x = self.conv_out(stack)

        x = self.decoder_1(x)
        x = torch.concat([x, f2], dim=1)      # (N, 2C, H/2, W/2)
        x = self.decoder_conv_1(x)

        x = self.decoder_2(x)
        x = torch.concat([x, f1], dim=1)
        x = self.decoder_conv_2(x)

        x = inp + x

        return x



















