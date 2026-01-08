"""
Visualization tools for Cross-Domain Degradation Transfer

Includes:
- Restoration result visualization
- t-SNE of degradation space
- Training curves
- Comparison figures for paper
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor from [-1, 1] to [0, 1] numpy array"""
    img = tensor.detach().cpu()
    img = (img + 1) / 2
    img = img.clamp(0, 1)

    if img.dim() == 4:
        img = img[0]  # Take first in batch

    img = img.permute(1, 2, 0).numpy()
    return img


def visualize_restoration(
    degraded: torch.Tensor,
    restored: torch.Tensor,
    clean: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Visualize restoration results

    Args:
        degraded: Degraded input image
        restored: Restored output image
        clean: Ground truth clean image (optional)
        save_path: Path to save figure
        title: Figure title

    Returns:
        matplotlib Figure
    """
    n_cols = 3 if clean is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(denormalize(degraded))
    axes[0].set_title('Degraded')
    axes[0].axis('off')

    axes[1].imshow(denormalize(restored))
    axes[1].set_title('Restored')
    axes[1].axis('off')

    if clean is not None:
        axes[2].imshow(denormalize(clean))
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_tsne_degradation(
    z_d_dict: Dict[str, torch.Tensor],
    deg_types: Optional[Dict[str, torch.Tensor]] = None,
    save_path: Optional[str] = None,
    perplexity: int = 30,
) -> Figure:
    """
    Plot t-SNE visualization of degradation representations

    Args:
        z_d_dict: Dictionary mapping domain names to degradation vectors [N, D]
        deg_types: Optional dictionary of degradation type labels [N]
        save_path: Path to save figure
        perplexity: t-SNE perplexity

    Returns:
        matplotlib Figure
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError("t-SNE requires: pip install scikit-learn")

    # Collect all z_d vectors
    all_z = []
    all_domains = []
    all_types = []

    for domain, z_d in z_d_dict.items():
        z = z_d.detach().cpu().numpy()
        all_z.append(z)
        all_domains.extend([domain] * len(z))

        if deg_types and domain in deg_types:
            types = deg_types[domain].detach().cpu().numpy()
            all_types.extend(types)
        else:
            all_types.extend([-1] * len(z))

    all_z = np.concatenate(all_z, axis=0)

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    z_2d = tsne.fit_transform(all_z)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot by domain
    domains = list(z_d_dict.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(domains)))

    for i, domain in enumerate(domains):
        mask = np.array(all_domains) == domain
        axes[0].scatter(z_2d[mask, 0], z_2d[mask, 1], c=[colors[i]], label=domain, alpha=0.6, s=20)

    axes[0].set_title('Degradation Space by Domain')
    axes[0].legend()
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')

    # Plot by degradation type
    unique_types = np.unique([t for t in all_types if t >= 0])
    type_names = ['noise', 'blur', 'artifact', 'compression', 'weather', 'other', 'mixed', 'unknown']

    if len(unique_types) > 0:
        type_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))

        for i, t in enumerate(unique_types):
            mask = np.array(all_types) == t
            label = type_names[int(t)] if int(t) < len(type_names) else f'Type {int(t)}'
            axes[1].scatter(z_2d[mask, 0], z_2d[mask, 1], c=[type_colors[i]], label=label, alpha=0.6, s=20)

        axes[1].legend()

    axes[1].set_title('Degradation Space by Type')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_curves(
    logs: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot training curves

    Args:
        logs: Dictionary of metric name -> list of values
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    n_metrics = len(logs)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, logs.items()):
        ax.plot(values)
        ax.set_title(name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_comparison_figure(
    results: Dict[str, Dict[str, torch.Tensor]],
    save_path: Optional[str] = None,
    methods: List[str] = None,
) -> Figure:
    """
    Create comparison figure for paper (multiple methods)

    Args:
        results: Nested dict: method -> {'degraded', 'restored', 'clean'}
        save_path: Path to save figure
        methods: Order of methods to display

    Returns:
        matplotlib Figure
    """
    if methods is None:
        methods = list(results.keys())

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods + 2, figsize=(3 * (n_methods + 2), 3))

    # Get first method's data for input/GT
    first_data = results[methods[0]]

    # Degraded input
    axes[0].imshow(denormalize(first_data['degraded']))
    axes[0].set_title('Input')
    axes[0].axis('off')

    # Method results
    for i, method in enumerate(methods):
        axes[i + 1].imshow(denormalize(results[method]['restored']))
        axes[i + 1].set_title(method)
        axes[i + 1].axis('off')

    # Ground truth
    if 'clean' in first_data:
        axes[-1].imshow(denormalize(first_data['clean']))
        axes[-1].set_title('GT')
        axes[-1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def save_result_grid(
    samples: List[Dict[str, torch.Tensor]],
    save_path: str,
    n_cols: int = 4,
) -> None:
    """
    Save a grid of restoration results

    Args:
        samples: List of dicts with 'degraded', 'restored', 'clean'
        save_path: Path to save figure
        n_cols: Number of columns (each showing degraded/restored/clean)
    """
    n_samples = min(len(samples), n_cols)
    fig, axes = plt.subplots(3, n_samples, figsize=(3 * n_samples, 9))

    row_titles = ['Degraded', 'Restored', 'Clean']

    for i in range(n_samples):
        sample = samples[i]

        axes[0, i].imshow(denormalize(sample['degraded']))
        axes[1, i].imshow(denormalize(sample['restored']))

        if 'clean' in sample:
            axes[2, i].imshow(denormalize(sample['clean']))

        for j in range(3):
            axes[j, i].axis('off')
            if i == 0:
                axes[j, i].set_ylabel(row_titles[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
