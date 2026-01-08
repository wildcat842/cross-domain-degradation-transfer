"""
Cross-Domain Degradation Transfer (CDDT) - Full Model

ICML 2026 Submission
"""

import torch
import torch.nn as nn

from .encoders import DegradationEncoder, ContentEncoder
from .decoder import CrossDomainDecoder


class CrossDomainDegradationTransfer(nn.Module):
    """
    전체 프레임워크
    """
    def __init__(self, deg_dim=256, content_dim=512, n_degradation_types=8):
        super().__init__()
        self.deg_encoder = DegradationEncoder(dim=deg_dim, n_degradation_types=n_degradation_types)
        self.content_encoder = ContentEncoder(dim=content_dim)
        self.decoder = CrossDomainDecoder(content_dim=content_dim, deg_dim=deg_dim)

        # Domain discriminator (adversarial training)
        self.domain_disc = nn.Sequential(
            nn.Linear(deg_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_degraded, x_clean=None):
        # 열화 이미지에서 표현 추출
        z_d, mu, logvar, deg_type = self.deg_encoder(x_degraded)
        z_c = self.content_encoder(x_degraded)

        # 복원 (열화 제거)
        x_restored = self.decoder(z_c, z_d, remove_degradation=True)

        return {
            'restored': x_restored,
            'z_d': z_d,
            'z_c': z_c,
            'mu': mu,
            'logvar': logvar,
            'deg_type': deg_type
        }

    def encode_degradation(self, x):
        """열화 표현만 추출"""
        z_d, mu, logvar, deg_type = self.deg_encoder(x)
        return z_d, deg_type

    def encode_content(self, x):
        """콘텐츠 표현만 추출"""
        return self.content_encoder(x)

    def restore(self, x_degraded):
        """이미지 복원"""
        return self.forward(x_degraded)['restored']

    def transfer_degradation(self, x_source, x_target):
        """
        소스 이미지의 열화를 타겟 이미지에 전이

        Args:
            x_source: 열화를 가져올 소스 이미지
            x_target: 열화를 적용할 타겟 이미지
        """
        z_d_source, _, _, _ = self.deg_encoder(x_source)
        z_c_target = self.content_encoder(x_target)

        # 열화 적용 (remove_degradation=False)
        return self.decoder(z_c_target, z_d_source, remove_degradation=False)
