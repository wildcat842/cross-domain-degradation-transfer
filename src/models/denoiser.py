"""
Simple denoising model (DnCNN-style) and NAFNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, DW_Expand: int = 2, FFN_Expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sg = SimpleGate()
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel: int = 3, width: int = 16, middle_blk_num: int = 1, enc_blk_nums: list = [], dec_blk_nums: list = []):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp: torch.Tensor) -> dict:
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp  # residual
        restored = x[:, :, :H, :W]
        noise = inp - restored  # estimated noise
        return {
            'restored': restored,
            'noise': noise,
        }

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)['restored']


class SimpleDenoiser(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        features: int = 64,
        depth: int = 8,
        use_batchnorm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        layers = [
            nn.Conv2d(channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> dict:
        noise = self.net(x)
        restored = x - noise if self.residual else noise
        return {
            'restored': restored,
            'noise': noise,
        }

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)['restored']
    def __init__(
        self,
        channels: int = 3,
        features: int = 64,
        depth: int = 8,
        use_batchnorm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        layers = [
            nn.Conv2d(channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> dict:
        noise = self.net(x)
        restored = x - noise if self.residual else noise
        return {
            'restored': restored,
            'noise': noise,
        }

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)['restored']