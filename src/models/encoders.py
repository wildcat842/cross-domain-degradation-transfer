"""
Encoders for Cross-Domain Degradation Transfer Learning

핵심 Contribution 1: Degradation-Content Disentanglement
"""

import torch
import torch.nn as nn


class DegradationEncoder(nn.Module):
    """
    도메인 불변(domain-invariant) 열화 표현 추출

    이론적 가정:
    - 열화 패턴은 저차원 매니폴드 Z_d에 존재
    - Z_d는 도메인 간 공유됨 (noise, blur, artifact 등)
    """
    def __init__(self, dim=256, n_degradation_types=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # 열화 표현 (continuous)
        self.mu = nn.Linear(256, dim)
        self.logvar = nn.Linear(256, dim)

        # 열화 유형 (discrete) - 해석 가능성
        self.type_classifier = nn.Linear(256, n_degradation_types)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_d = mu + eps * std

        # 열화 유형 (보조 출력)
        deg_type = self.type_classifier(h)

        return z_d, mu, logvar, deg_type


class ContentEncoder(nn.Module):
    """
    열화 불변(degradation-invariant) 콘텐츠 표현 추출

    이론적 가정:
    - 콘텐츠는 열화와 독립: I(z_c; z_d) ≈ 0
    - 도메인 특화: z_c는 도메인마다 다른 공간
    """
    def __init__(self, dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, dim, 3, 2, 1),
        )

    def forward(self, x):
        return self.encoder(x)
