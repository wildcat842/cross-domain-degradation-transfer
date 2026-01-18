"""
Evaluation script for Cross-Domain / Cross-Corruption Degradation Transfer

Modes:
- single        : single domain evaluation
- cross_domain  : single source -> target evaluation
- imagenet-all  : ImageNet corruption transfer (noise/blur/weather)
- all           : general cross-domain transfer (imagenet/ldct/dibco/fmd)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import yaml
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CrossDomainDegradationTransfer
from src.data import create_multi_domain_loader, create_cross_domain_pairs
from src.utils import compute_psnr, compute_ssim, MetricTracker


# -------------------------------------------------
# Domain groups
# -------------------------------------------------
IMAGENET_CORRUPTION_DOMAINS = [
    'imagenet-noise',
    'imagenet-blur',
    'imagenet-weather',
]

CROSS_DOMAIN_DOMAINS = [
    'imagenet',
    'ldct',
    'dibco',
    'fmd',
]


# -------------------------------------------------
# Argument parsing
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CDDT model')

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)

    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')

    parser.add_argument(
        '--mode',
        type=str,
        default='cross_domain',
        choices=['single', 'cross_domain', 'imagenet-all', 'all'],
    )

    parser.add_argument('--source_domain', type=str, default=None)
    parser.add_argument('--target_domain', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_shots', type=int, default=0)

    return parser.parse_args()


# -------------------------------------------------
# Model loading
# -------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    model_cfg = config.get('model', {})
    model = CrossDomainDegradationTransfer(
        deg_dim=model_cfg.get('deg_dim', 256),
        content_dim=model_cfg.get('content_dim', 512),
        n_degradation_types=model_cfg.get('n_degradation_types', 8),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


# -------------------------------------------------
# Evaluation core
# -------------------------------------------------
@torch.no_grad()
def evaluate_domain(model, loader, device):
    tracker = MetricTracker(['psnr', 'ssim'])

    for batch in loader:
        degraded = batch['degraded'].to(device)
        clean = batch['clean'].to(device)

        out = model(degraded)
        restored = out['restored'] if isinstance(out, dict) else out

        restored = (restored + 1) / 2
        target = (clean + 1) / 2

        tracker.update('psnr', compute_psnr(restored, target))
        tracker.update('ssim', compute_ssim(restored, target))

    return tracker.compute()


# -------------------------------------------------
# Matrix evaluation
# -------------------------------------------------
def evaluate_matrix(model, domains, data_root, device, batch_size, n_shots):
    rows = []

    for src in domains:
        for tgt in domains:
            if src == tgt:
                continue

            print(f"Evaluating {src} -> {tgt}")

            _, _, test_loader = create_cross_domain_pairs(
                source_domain=src,
                target_domain=tgt,
                data_root=data_root,
                n_shots=n_shots,
                batch_size=batch_size,
            )

            if len(test_loader) == 0:
                continue

            metrics = evaluate_domain(model, test_loader, device)

            rows.append({
                'source': src,
                'target': tgt,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
            })

            print(f"  PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")

    return pd.DataFrame(rows)


# -------------------------------------------------
# Heatmap plotting
# -------------------------------------------------
def plot_heatmap(df: pd.DataFrame, metric: str, save_path: Path, title: str):
    pivot = df.pivot(index='source', columns='target', values=metric)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar=True,
        square=True,
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()

    # Load config overrides
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)

    # ------------------------------
    # Mode handling
    # ------------------------------
    if args.mode == 'imagenet-all':
        domains = IMAGENET_CORRUPTION_DOMAINS
        tag = "imagenet_cross_corruption"

    elif args.mode == 'all':
        domains = CROSS_DOMAIN_DOMAINS
        tag = "cross_domain"

    elif args.mode == 'cross_domain':
        domains = [args.source_domain, args.target_domain]
        tag = f"{args.source_domain}_to_{args.target_domain}"

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    df = evaluate_matrix(
        model,
        domains,
        args.data_root,
        device,
        args.batch_size,
        args.n_shots,
    )

    csv_path = output_dir / f"{tag}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # Heatmaps
    plot_heatmap(
        df,
        metric='psnr',
        save_path=output_dir / f"{tag}_psnr_heatmap.png",
        title=f"{tag} PSNR",
    )

    plot_heatmap(
        df,
        metric='ssim',
        save_path=output_dir / f"{tag}_ssim_heatmap.png",
        title=f"{tag} SSIM",
    )

    print(f"Heatmaps saved to {output_dir}")


if __name__ == '__main__':
    main()
