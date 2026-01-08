"""
Evaluation metrics for image restoration

Metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
"""

import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import numpy as np


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Compute PSNR (Peak Signal-to-Noise Ratio)

    Args:
        pred: Predicted image [B, C, H, W] in range [0, data_range]
        target: Target image [B, C, H, W] in range [0, data_range]
        data_range: Maximum value of the data (1.0 for normalized, 255 for uint8)

    Returns:
        PSNR value (higher is better)
    """
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(data_range ** 2 / (mse + 1e-8))
    return psnr.mean()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Compute SSIM (Structural Similarity Index)

    Args:
        pred: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]
        window_size: Size of the Gaussian window
        data_range: Maximum value of the data

    Returns:
        SSIM value in range [-1, 1] (higher is better)
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Create Gaussian window
    window = _create_gaussian_window(window_size, pred.shape[1]).to(pred.device)

    # Compute means
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=pred.shape[1])
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=target.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=target.shape[1]) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=pred.shape[1]) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def _create_gaussian_window(window_size: int, channels: int) -> torch.Tensor:
    """Create a Gaussian window for SSIM computation"""
    sigma = 1.5

    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    window = g.outer(g)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(channels, 1, window_size, window_size).contiguous()

    return window


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = 'alex',
    lpips_model: Optional[object] = None,
) -> torch.Tensor:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity)

    Requires: pip install lpips

    Args:
        pred: Predicted image [B, C, H, W] in range [-1, 1]
        target: Target image [B, C, H, W] in range [-1, 1]
        net: Network to use ('alex', 'vgg', 'squeeze')
        lpips_model: Pre-initialized LPIPS model (optional)

    Returns:
        LPIPS value (lower is better)
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("LPIPS requires: pip install lpips")

    if lpips_model is None:
        lpips_model = lpips.LPIPS(net=net).to(pred.device)
        lpips_model.eval()

    with torch.no_grad():
        distance = lpips_model(pred, target)

    return distance.mean()


class MetricTracker:
    """
    Track and aggregate metrics during training/evaluation

    Usage:
        tracker = MetricTracker(['psnr', 'ssim'])
        for batch in dataloader:
            tracker.update('psnr', compute_psnr(pred, target))
            tracker.update('ssim', compute_ssim(pred, target))
        results = tracker.compute()
    """

    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.values: Dict[str, List[float]] = {m: [] for m in metrics}

    def update(self, name: str, value: Union[float, torch.Tensor]):
        """Add a metric value"""
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.values[name].append(value)

    def compute(self) -> Dict[str, float]:
        """Compute mean of all metrics"""
        return {name: np.mean(values) for name, values in self.values.items() if values}

    def reset(self):
        """Reset all tracked values"""
        self.values = {m: [] for m in self.metrics}

    def __str__(self) -> str:
        results = self.compute()
        return ' | '.join([f'{k}: {v:.4f}' for k, v in results.items()])


def evaluate_restoration(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = torch.device('cuda'),
    metrics: List[str] = ['psnr', 'ssim'],
) -> Dict[str, float]:
    """
    Evaluate restoration model on a dataset

    Args:
        model: Restoration model
        dataloader: DataLoader with 'degraded' and 'clean' keys
        device: Device to run evaluation on
        metrics: List of metrics to compute

    Returns:
        Dictionary of metric name -> value
    """
    model.eval()
    tracker = MetricTracker(metrics)

    with torch.no_grad():
        for batch in dataloader:
            degraded = batch['degraded'].to(device)
            clean = batch['clean'].to(device)

            # Get restoration
            outputs = model(degraded)
            restored = outputs['restored'] if isinstance(outputs, dict) else outputs

            # Denormalize from [-1, 1] to [0, 1]
            restored = (restored + 1) / 2
            clean = (clean + 1) / 2

            # Compute metrics
            if 'psnr' in metrics:
                tracker.update('psnr', compute_psnr(restored, clean))
            if 'ssim' in metrics:
                tracker.update('ssim', compute_ssim(restored, clean))

    return tracker.compute()
