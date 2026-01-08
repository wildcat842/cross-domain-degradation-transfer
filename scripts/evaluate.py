"""
Evaluation script for Cross-Domain Degradation Transfer Learning

Supports:
- Single domain evaluation
- Cross-domain zero-shot evaluation
- Few-shot adaptation evaluation
- Comparison with baselines
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import yaml
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CrossDomainDegradationTransfer
from src.data import create_multi_domain_loader, create_cross_domain_pairs
from src.utils import (
    compute_psnr, compute_ssim, MetricTracker,
    visualize_restoration, plot_tsne_degradation, save_result_grid
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CDDT model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (uses checkpoint config if not specified)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')

    # Evaluation mode
    parser.add_argument('--mode', type=str, default='cross_domain',
                        choices=['single', 'cross_domain', 'all'],
                        help='Evaluation mode')
    parser.add_argument('--source_domain', type=str, default='imagenet',
                        help='Source domain (for cross_domain mode)')
    parser.add_argument('--target_domain', type=str, default='ldct',
                        help='Target domain (for cross_domain/single mode)')

    # Options
    parser.add_argument('--save_images', action='store_true',
                        help='Save restoration results')
    parser.add_argument('--visualize_tsne', action='store_true',
                        help='Generate t-SNE visualization')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_vis_samples', type=int, default=10,
                        help='Number of samples to visualize')

    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    model = CrossDomainDegradationTransfer(
        deg_dim=model_config.get('deg_dim', 256),
        content_dim=model_config.get('content_dim', 512),
        n_degradation_types=model_config.get('n_degradation_types', 8),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config


def evaluate_domain(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    domain_name: str = '',
    collect_embeddings: bool = False,
) -> Dict:
    """Evaluate model on a single domain"""
    tracker = MetricTracker(['psnr', 'ssim'])

    all_z_d = []
    all_deg_types = []
    samples = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Evaluating {domain_name}'):
            degraded = batch['degraded'].to(device)
            clean = batch['clean'].to(device)

            outputs = model(degraded)

            # Denormalize
            restored = (outputs['restored'] + 1) / 2
            target = (clean + 1) / 2

            # Compute metrics
            psnr = compute_psnr(restored, target)
            ssim = compute_ssim(restored, target)

            tracker.update('psnr', psnr)
            tracker.update('ssim', ssim)

            # Collect embeddings for t-SNE
            if collect_embeddings:
                all_z_d.append(outputs['z_d'].cpu())
                all_deg_types.append(outputs['deg_type'].argmax(dim=1).cpu())

            # Save some samples for visualization
            if len(samples) < 10:
                samples.append({
                    'degraded': degraded[0].cpu(),
                    'restored': outputs['restored'][0].cpu(),
                    'clean': clean[0].cpu(),
                })

    results = tracker.compute()

    if collect_embeddings:
        results['z_d'] = torch.cat(all_z_d, dim=0)
        results['deg_types'] = torch.cat(all_deg_types, dim=0)

    results['samples'] = samples

    return results


def evaluate_cross_domain_matrix(
    model: torch.nn.Module,
    data_root: str,
    device: torch.device,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Evaluate cross-domain transfer performance matrix

    Returns DataFrame with Source -> Target PSNR values
    """
    domains = ['imagenet', 'ldct', 'dibco', 'fmd']
    results = []

    for source in domains:
        for target in domains:
            if source == target:
                continue

            print(f"\nEvaluating: {source} -> {target}")

            # Load target test data
            _, _, test_loader = create_cross_domain_pairs(
                source_domain=source,
                target_domain=target,
                data_root=data_root,
                n_shots=0,
                batch_size=batch_size,
            )

            if len(test_loader) == 0:
                print(f"  No test data for {target}")
                continue

            # Evaluate
            domain_results = evaluate_domain(model, test_loader, device, f'{source}->{target}')

            results.append({
                'source': source,
                'target': target,
                'psnr': domain_results['psnr'],
                'ssim': domain_results['ssim'],
            })

            print(f"  PSNR: {domain_results['psnr']:.2f}, SSIM: {domain_results['ssim']:.4f}")

    df = pd.DataFrame(results)
    return df


def main():
    args = parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)

    if args.mode == 'all':
        # Full cross-domain evaluation matrix
        print("\n" + "=" * 50)
        print("Cross-Domain Transfer Evaluation Matrix")
        print("=" * 50)

        df = evaluate_cross_domain_matrix(model, args.data_root, device, args.batch_size)

        # Save results
        df.to_csv(output_dir / 'cross_domain_results.csv', index=False)

        # Print pivot table
        pivot = df.pivot(index='source', columns='target', values='psnr')
        print("\nPSNR Matrix (Source -> Target):")
        print(pivot.to_string())

        # Calculate averages
        avg_psnr = df['psnr'].mean()
        avg_ssim = df['ssim'].mean()
        print(f"\nAverage PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")

    elif args.mode == 'cross_domain':
        # Single cross-domain evaluation
        print(f"\nEvaluating: {args.source_domain} -> {args.target_domain}")

        _, _, test_loader = create_cross_domain_pairs(
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            data_root=args.data_root,
            n_shots=0,
            batch_size=args.batch_size,
        )

        results = evaluate_domain(
            model, test_loader, device,
            f'{args.source_domain}->{args.target_domain}',
            collect_embeddings=args.visualize_tsne
        )

        print(f"PSNR: {results['psnr']:.2f}")
        print(f"SSIM: {results['ssim']:.4f}")

        # Save visualizations
        if args.save_images and results['samples']:
            save_result_grid(
                results['samples'][:args.n_vis_samples],
                str(output_dir / f'{args.source_domain}_to_{args.target_domain}_samples.png')
            )

    elif args.mode == 'single':
        # Single domain evaluation
        print(f"\nEvaluating on: {args.target_domain}")

        loaders = create_multi_domain_loader(
            domains=[args.target_domain],
            data_root=args.data_root,
            batch_size=args.batch_size,
            split='test',
        )

        if args.target_domain not in loaders:
            print(f"Error: Could not load {args.target_domain} dataset")
            return

        results = evaluate_domain(
            model, loaders[args.target_domain], device,
            args.target_domain,
            collect_embeddings=args.visualize_tsne
        )

        print(f"PSNR: {results['psnr']:.2f}")
        print(f"SSIM: {results['ssim']:.4f}")

    # Generate t-SNE visualization
    if args.visualize_tsne:
        print("\nGenerating t-SNE visualization...")

        # Collect embeddings from all domains
        domains = ['imagenet', 'ldct', 'dibco', 'fmd']
        z_d_dict = {}
        deg_types_dict = {}

        for domain in domains:
            loaders = create_multi_domain_loader(
                domains=[domain],
                data_root=args.data_root,
                batch_size=args.batch_size,
                split='test',
            )

            if domain not in loaders:
                continue

            results = evaluate_domain(
                model, loaders[domain], device, domain,
                collect_embeddings=True
            )

            if 'z_d' in results:
                z_d_dict[domain] = results['z_d'][:500]  # Limit samples
                deg_types_dict[domain] = results['deg_types'][:500]

        if z_d_dict:
            plot_tsne_degradation(
                z_d_dict,
                deg_types_dict,
                save_path=str(output_dir / 'tsne_degradation_space.png')
            )
            print(f"t-SNE saved to: {output_dir / 'tsne_degradation_space.png'}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
