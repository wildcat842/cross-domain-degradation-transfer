"""
Simple denoising model (DnCNN-style).
"""

import torch
import torch.nn as nn


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