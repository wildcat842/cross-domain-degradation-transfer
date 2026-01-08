"""
Decoder for Cross-Domain Degradation Transfer Learning

핵심 Contribution 2: Cross-Domain Transfer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    def __init__(self, feat_dim, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(feat_dim)
        self.gamma = nn.Linear(style_dim, feat_dim)
        self.beta = nn.Linear(style_dim, feat_dim)

    def forward(self, x, style):
        x = self.norm(x)
        gamma = self.gamma(style).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(style).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class CrossDomainDecoder(nn.Module):
    """
    콘텐츠 + 열화 표현 → 복원 이미지

    핵심: z_d를 "제거"하여 clean 이미지 생성
    """
    def __init__(self, content_dim=512, deg_dim=256):
        super().__init__()

        # 동적 채널 계산: content_dim에서 시작하여 절반씩 감소
        ch1 = content_dim
        ch2 = max(content_dim // 2, 64)
        ch3 = max(content_dim // 4, 32)
        ch_out = max(content_dim // 8, 16)

        # 열화 제거를 위한 adaptive normalization
        self.adain_layers = nn.ModuleList([
            AdaIN(ch1, deg_dim),
            AdaIN(ch2, deg_dim),
            AdaIN(ch3, deg_dim),
        ])

        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(ch1, ch2, 4, 2, 1),
            nn.ConvTranspose2d(ch2, ch3, 4, 2, 1),
            nn.ConvTranspose2d(ch3, ch_out, 4, 2, 1),
        ])

        self.output = nn.Conv2d(ch_out, 3, 3, 1, 1)

    def forward(self, z_c, z_d, remove_degradation=True):
        """
        Args:
            z_c: 콘텐츠 표현 [B, 512, H, W]
            z_d: 열화 표현 [B, 256]
            remove_degradation: True면 열화 제거 (복원), False면 열화 적용
        """
        x = z_c

        # 열화 제거: z_d의 "반대" 방향으로 이동
        if remove_degradation:
            z_d_inv = -z_d  # 단순 반전 (실제로는 학습된 변환)
        else:
            z_d_inv = z_d

        for i, (adain, dec) in enumerate(zip(self.adain_layers, self.decoder)):
            x = adain(x, z_d_inv)
            x = F.relu(dec(x))

        return torch.tanh(self.output(x))
