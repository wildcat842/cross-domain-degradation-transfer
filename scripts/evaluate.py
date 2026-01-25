"""
Evaluation script for Cross-Domain / Cross-Corruption Degradation Transfer

Modes:
- single        : single domain evaluation
- cross_domain  : single source -> target evaluation
- imagenet-all  : ImageNet corruption transfer (noise/blur/weather)  [SOURCE-CKPT PER ROW]
- all           : general cross-domain transfer (imagenet/ldct/dibco/fmd)
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CrossDomainDegradationTransfer
from src.data import create_cross_domain_pairs
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
    # NOTE: imagenet-all에서는 config['checkpoints']를 쓰므로 checkpoint는 선택적으로 둠
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Single checkpoint path (used for cross_domain/all unless overridden by config)",
                        required=False)
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
# Matrix evaluation (single model for all pairs)
# -------------------------------------------------
def evaluate_matrix(model, domains, data_root, device, batch_size, n_shots, image_size):
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
                image_size=image_size,
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
# Matrix evaluation (per-source checkpoint)
# -------------------------------------------------
def evaluate_matrix_per_source_ckpt(ckpt_dict, domains, data_root, device, batch_size, n_shots, image_size):
    """
    Each row (source) uses its own checkpoint.
    ckpt_dict example:
      {
        "imagenet-noise": ".../noise_ckpt.pth",
        "imagenet-blur": ".../blur_ckpt.pth",
        "imagenet-weather": ".../weather_ckpt.pth"
      }
    """
    rows = []

    for src in domains:
        if src not in ckpt_dict:
            raise ValueError(f"[imagenet-all] Missing checkpoint for source domain: {src}")

        ckpt_path = ckpt_dict[src]
        print(f"\n[Load] source={src} checkpoint={ckpt_path}")
        model = load_model(ckpt_path, device)

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
                image_size=image_size,
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

    fmt = ".2f" if metric == "psnr" else ".4f"
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
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

    cfg = {}  # IMPORTANT: avoid UnboundLocalError when args.config is None
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Mode handling
    # ------------------------------
    if args.mode == 'imagenet-all':
        if 'checkpoints' not in cfg or not isinstance(cfg['checkpoints'], dict):
            raise ValueError(
                "For mode=imagenet-all, config must define 'checkpoints' dict: {source_domain: ckpt_path}"
            )
        ckpt_dict = cfg['checkpoints']
        domains = IMAGENET_CORRUPTION_DOMAINS
        tag = "imagenet_cross_corruption"

        df = evaluate_matrix_per_source_ckpt(
            ckpt_dict=ckpt_dict,
            domains=domains,
            data_root=args.data_root,
            device=device,
            batch_size=args.batch_size,
            n_shots=args.n_shots,
            image_size=cfg.get('image_size', 64),
        )

    elif args.mode == 'all':
        if args.checkpoint is None:
            raise ValueError("For mode=all, you must provide --checkpoint (or set checkpoint in config and pass --config).")
        domains = CROSS_DOMAIN_DOMAINS
        tag = "cross_domain"

        model = load_model(args.checkpoint, device)
        df = evaluate_matrix(
            model,
            domains,
            args.data_root,
            device,
            args.batch_size,
            args.n_shots,
            cfg.get('image_size', 64),
        )

    elif args.mode == 'cross_domain':
        if args.checkpoint is None:
            raise ValueError("For mode=cross_domain, you must provide --checkpoint (or set checkpoint in config and pass --config).")
        if args.source_domain is None or args.target_domain is None:
            raise ValueError("For mode=cross_domain, you must provide --source_domain and --target_domain.")
        domains = [args.source_domain, args.target_domain]
        tag = f"{args.source_domain}_to_{args.target_domain}"

        model = load_model(args.checkpoint, device)
        df = evaluate_matrix(
            model,
            domains,
            args.data_root,
            device,
            args.batch_size,
            args.n_shots,
            cfg.get('image_size', 64),
        )

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    # Save CSV + heatmaps
    csv_path = output_dir / f"{tag}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

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
