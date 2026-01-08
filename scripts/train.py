"""
Training script for Cross-Domain Degradation Transfer Learning
"""

import argparse
import sys
sys.path.insert(0, '..')

import torch
from torch.utils.data import DataLoader

from src.models import CrossDomainDegradationTransfer
from src.losses import ICMLLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train CDDT model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize model
    model = CrossDomainDegradationTransfer()
    criterion = ICMLLoss()

    # TODO: Add data loading
    # TODO: Add training loop
    # TODO: Add logging and checkpointing

    print("Training script initialized")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == '__main__':
    main()
