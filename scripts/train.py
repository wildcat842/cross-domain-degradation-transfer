"""
Training script for Cross-Domain Degradation Transfer Learning

Supports:
- Single domain training
- Multi-domain joint training
- Cross-domain transfer with few-shot adaptation
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CrossDomainDegradationTransfer, SimpleDenoiser
from src.losses import ICMLLoss
from src.data import create_multi_domain_loader, create_cross_domain_pairs
from src.data.loader import MultiDomainIterator
from src.utils import MetricTracker, compute_psnr, compute_ssim


def parse_args():
    parser = argparse.ArgumentParser(description='Train CDDT model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--model', type=str, default='cddt',
                        choices=['cddt', 'denoiser'],
                        help='Model to train')

    # Cross-domain transfer settings
    parser.add_argument('--source_domain', type=str, default=None,
                        help='Source domain for transfer')
    parser.add_argument('--target_domain', type=str, default=None,
                        help='Target domain for transfer')
    parser.add_argument('--n_shots', type=int, default=0,
                        help='Number of target samples for few-shot (0=zero-shot)')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(
    model: nn.Module,
    iterator: MultiDomainIterator,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    model_type: str,
    writer: SummaryWriter = None,
) -> dict:
    """Train for one epoch with multi-domain data"""
    model.train()
    if model_type == 'denoiser':
        tracker = MetricTracker(['loss', 'psnr'])
    else:
        tracker = MetricTracker(['loss', 'recon', 'kl', 'disentangle', 'psnr'])

    n_iters = config['training'].get('iters_per_epoch', 1000)
    pbar = tqdm(range(n_iters), desc=f'Epoch {epoch}')

    for i in pbar:
        domain, batch = next(iterator)

        degraded = batch['degraded'].to(device)
        clean = batch['clean'].to(device)

        # Forward
        outputs = model(degraded)

        # Compute loss
        if model_type == 'denoiser':
            restored = outputs['restored'] if isinstance(outputs, dict) else outputs
            total_loss = F.mse_loss(restored, clean)
            losses = {'total': total_loss}
        else:
            losses = criterion(outputs, clean)

        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # Track metrics
        tracker.update('loss', losses['total'])
        if model_type != 'denoiser':
            tracker.update('recon', losses['recon'])
            tracker.update('kl', losses['kl'])
            tracker.update('disentangle', losses['disentangle'])

        # Compute PSNR
        with torch.no_grad():
            restored = outputs['restored'] if isinstance(outputs, dict) else outputs
            restored = (restored + 1) / 2
            target = (clean + 1) / 2
            psnr = compute_psnr(restored, target)
            tracker.update('psnr', psnr)

        pbar.set_postfix({'loss': f"{losses['total'].item():.4f}", 'psnr': f"{psnr.item():.2f}"})

        # Log to tensorboard
        if writer and i % config['experiment'].get('log_interval', 100) == 0:
            step = epoch * n_iters + i
            writer.add_scalar('train/loss', losses['total'].item(), step)
            writer.add_scalar('train/psnr', psnr.item(), step)
            if model_type != 'denoiser':
                writer.add_scalar('train/domain', hash(domain) % 100, step)

    return tracker.compute()


def validate(
    model: nn.Module,
    loaders: dict,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
) -> dict:
    """Validate on all domains"""
    model.eval()
    results = {}

    with torch.no_grad():
        for domain, loader in loaders.items():
            tracker = MetricTracker(['psnr', 'ssim'])

            for batch in loader:
                degraded = batch['degraded'].to(device)
                clean = batch['clean'].to(device)

                outputs = model(degraded)
                restored = outputs['restored'] if isinstance(outputs, dict) else outputs
                restored = (restored + 1) / 2
                target = (clean + 1) / 2

                tracker.update('psnr', compute_psnr(restored, target))
                tracker.update('ssim', compute_ssim(restored, target))

            results[domain] = tracker.compute()

    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    config: dict,
    save_path: str,
    best: bool = False,
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    torch.save(checkpoint, save_path)

    if best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup experiment directory
    exp_name = args.exp_name or config['experiment'].get('name', 'cddt')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(config['experiment']['save_dir']) / f'{exp_name}_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    if args.model == 'denoiser':
        model = SimpleDenoiser().to(device)
    else:
        model = CrossDomainDegradationTransfer(
            deg_dim=config['model']['deg_dim'],
            content_dim=config['model']['content_dim'],
            n_degradation_types=config['model']['n_degradation_types'],
        ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    if args.model == 'denoiser':
        criterion = None
    else:
        criterion = ICMLLoss(
            recon_weight=config['loss']['recon_weight'],
            kl_weight=config['loss']['kl_weight'],
            disentangle_weight=config['loss']['disentangle_weight'],
            domain_weight=config['loss']['domain_weight'],
        )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
    )

    # Data loaders
    if args.source_domain and args.target_domain:
        # Cross-domain transfer mode
        print(f"Cross-domain transfer: {args.source_domain} -> {args.target_domain}")
        print(f"Few-shot samples: {args.n_shots}")

        source_loader, fewshot_loader, test_loader = create_cross_domain_pairs(
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            data_root=config['data']['train_dir'].replace('/train', ''),
            n_shots=args.n_shots,
            batch_size=config['training']['batch_size'],
            image_size=config['data']['image_size'],
            num_workers=config['data']['num_workers'],
        )
        train_loaders = {args.source_domain: source_loader}
        val_loaders = {args.target_domain: test_loader}
    else:
        # Multi-domain training mode
        domains = config.get('domains', ['imagenet', 'ldct', 'dibco', 'fmd'])
        print(f"Multi-domain training: {domains}")

        train_loaders = create_multi_domain_loader(
            domains=domains,
            data_root=config['data']['train_dir'].replace('/train', ''),
            batch_size=config['training']['batch_size'],
            image_size=config['data']['image_size'],
            num_workers=config['data']['num_workers'],
            split='train',
        )
        val_loaders = create_multi_domain_loader(
            domains=domains,
            data_root=config['data']['val_dir'].replace('/val', ''),
            batch_size=config['training']['batch_size'],
            image_size=config['data']['image_size'],
            num_workers=config['data']['num_workers'],
            split='val',
        )

    # Multi-domain iterator
    try:
        iterator = MultiDomainIterator(train_loaders)
    except (ValueError, FileNotFoundError) as e:
        print(f"Data loader error: {e}")
        sys.exit(1)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Tensorboard
    writer = SummaryWriter(exp_dir / 'logs')

    # Training loop
    best_psnr = 0.0

    for epoch in range(start_epoch, config['training']['epochs']):
        # Train
        train_metrics = train_epoch(
            model, iterator, criterion, optimizer, device, epoch, config, args.model, writer
        )
        print(f"Epoch {epoch} - Train: {train_metrics}")

        # Validate
        val_results = validate(model, val_loaders, criterion, device, args.model)
        print(f"Epoch {epoch} - Val: {val_results}")

        # Log validation results
        avg_psnr = sum(r['psnr'] for r in val_results.values()) / len(val_results)
        writer.add_scalar('val/avg_psnr', avg_psnr, epoch)

        for domain, metrics in val_results.items():
            writer.add_scalar(f'val/{domain}_psnr', metrics['psnr'], epoch)
            writer.add_scalar(f'val/{domain}_ssim', metrics['ssim'], epoch)

        # Save checkpoint
        is_best = avg_psnr > best_psnr
        if is_best:
            best_psnr = avg_psnr

        if (epoch + 1) % config['experiment'].get('save_interval', 10) == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, config,
                str(exp_dir / f'checkpoint_{epoch:04d}.pth'),
                best=is_best
            )

        scheduler.step()

    writer.close()
    print(f"Training complete. Best PSNR: {best_psnr:.2f}")
    print(f"Results saved to: {exp_dir}")


if __name__ == '__main__':
    main()
