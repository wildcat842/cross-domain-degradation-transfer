"""
ICML 2026 Loss Functions

이론적 동기 부여가 포함된 손실 함수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICMLLoss(nn.Module):
    """
    ICML 논문용 손실 함수 (이론적 동기 부여 포함)
    """
    def __init__(self,
                 recon_weight=1.0,
                 kl_weight=0.01,
                 disentangle_weight=0.1,
                 domain_weight=0.1):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.disentangle_weight = disentangle_weight
        self.domain_weight = domain_weight

    def forward(self, outputs, targets, domain_labels=None):
        losses = {}

        # 1. Reconstruction Loss
        losses['recon'] = F.l1_loss(outputs['restored'], targets) * self.recon_weight

        # 2. KL Divergence (VAE regularization for z_d)
        # 이론적 동기: z_d가 compact한 매니폴드에 있도록
        mu, logvar = outputs['mu'], outputs['logvar']
        losses['kl'] = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * self.kl_weight

        # 3. Disentanglement Loss (z_c와 z_d의 독립성)
        # 이론적 동기: I(z_c; z_d) ≈ 0
        z_c_flat = outputs['z_c'].flatten(2).mean(dim=2)  # [B, 512]
        z_d = outputs['z_d']  # [B, 256]

        # HSIC (Hilbert-Schmidt Independence Criterion) 근사
        losses['disentangle'] = self._hsic(z_c_flat, z_d) * self.disentangle_weight

        # 4. Domain Adversarial Loss (z_d가 도메인 불변)
        # 이론적 동기: 열화 표현이 도메인에 독립
        if domain_labels is not None:
            # Gradient reversal for domain-invariant z_d
            losses['domain'] = F.binary_cross_entropy_with_logits(
                outputs['domain_pred'], domain_labels
            ) * self.domain_weight

        losses['total'] = sum(losses.values())
        return losses

    def _hsic(self, x, y):
        """
        HSIC: 두 변수의 독립성 측정
        낮을수록 더 독립적
        """
        n = x.size(0)

        # RBF kernel
        def rbf_kernel(a, b, sigma=1.0):
            dist = torch.cdist(a, b, p=2)
            return torch.exp(-dist ** 2 / (2 * sigma ** 2))

        Kx = rbf_kernel(x, x)
        Ky = rbf_kernel(y, y)

        # Centering
        H = torch.eye(n, device=x.device) - 1.0 / n
        Kxc = H @ Kx @ H
        Kyc = H @ Ky @ H

        # HSIC
        hsic = torch.trace(Kxc @ Kyc) / (n - 1) ** 2
        return hsic
