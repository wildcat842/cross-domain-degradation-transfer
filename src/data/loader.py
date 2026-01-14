"""
Multi-domain data loader for cross-domain transfer experiments
"""

from typing import Dict, List, Optional, Tuple
import random

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T

from .datasets import get_dataset


def get_default_transform(image_size: int = 256, train: bool = True):
    """Get default image transforms"""
    if train:
        return T.Compose([
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return T.Compose([
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


def create_multi_domain_loader(
    domains: List[str] = ['imagenet', 'ldct', 'dibco', 'fmd'],
    data_root: str = './data',
    batch_size: int = 16,
    image_size: int = 256,
    num_workers: int = 4,
    split: str = 'train',
) -> Dict[str, DataLoader]:
    """
    Create data loaders for multiple domains

    Args:
        domains: List of domain names
        data_root: Root directory containing all datasets
        batch_size: Batch size for each loader
        image_size: Image size for cropping
        num_workers: Number of data loading workers
        split: 'train', 'val', or 'test'

    Returns:
        Dictionary mapping domain names to DataLoaders
    """
    loaders = {}
    transform = get_default_transform(image_size, train=(split == 'train'))

    domain_roots = {
        # ImageNet-C base
        'imagenet': f'{data_root}/imagenet-c',
        'imagenet-c': f'{data_root}/imagenet-c',
        # ImageNet-C corruption categories (all use same root)
        'imagenet-noise': f'{data_root}/imagenet-c',
        'imagenet-blur': f'{data_root}/imagenet-c',
        'imagenet-weather': f'{data_root}/imagenet-c',
        'imagenet-digital': f'{data_root}/imagenet-c',
        # Other domains
        'ldct': f'{data_root}/ldct',
        'dibco': f'{data_root}/dibco',
        'fmd': f'{data_root}/fmd',
    }

    for domain in domains:
        root = domain_roots.get(domain, f'{data_root}/{domain}')

        try:
            dataset = get_dataset(domain, root, split, transform)

            if len(dataset) > 0:
                # For small datasets, adjust batch_size and drop_last
                effective_batch_size = min(batch_size, len(dataset))
                # Only drop_last if we have enough samples for at least 2 batches
                effective_drop_last = (split == 'train') and (len(dataset) >= 2 * effective_batch_size)

                loaders[domain] = DataLoader(
                    dataset,
                    batch_size=effective_batch_size,
                    shuffle=(split == 'train'),
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=effective_drop_last,
                )
                print(f"[{domain}] Loaded {len(dataset)} samples (batch_size={effective_batch_size})")
            else:
                print(f"[{domain}] Warning: No samples found at {root}")

        except Exception as e:
            print(f"[{domain}] Error loading dataset: {e}")

    if len(loaders) == 0:
        checked = ", ".join([f"{d} -> {domain_roots.get(d, f'{data_root}/{d}')}" for d in domains])
        raise FileNotFoundError(
            f"No datasets found for requested domains {domains}. Checked: {checked}. "
            "Please verify the dataset directories or set the correct 'data_root' in your config."
        )

    return loaders


def create_cross_domain_pairs(
    source_domain: str,
    target_domain: str,
    data_root: str = './data',
    n_shots: int = 0,
    batch_size: int = 16,
    image_size: int = 256,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for cross-domain transfer experiment

    Args:
        source_domain: Source domain for training
        target_domain: Target domain for evaluation
        data_root: Root directory containing all datasets
        n_shots: Number of target domain samples for few-shot (0 = zero-shot)
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of workers

    Returns:
        Tuple of (source_train_loader, target_fewshot_loader, target_test_loader)
    """
    train_transform = get_default_transform(image_size, train=True)
    test_transform = get_default_transform(image_size, train=False)

    domain_roots = {
        # ImageNet-C base
        'imagenet': f'{data_root}/imagenet-c',
        'imagenet-c': f'{data_root}/imagenet-c',
        # ImageNet-C corruption categories (all use same root)
        'imagenet-noise': f'{data_root}/imagenet-c',
        'imagenet-blur': f'{data_root}/imagenet-c',
        'imagenet-weather': f'{data_root}/imagenet-c',
        'imagenet-digital': f'{data_root}/imagenet-c',
        # Other domains
        'ldct': f'{data_root}/ldct',
        'dibco': f'{data_root}/dibco',
        'fmd': f'{data_root}/fmd',
    }

    # Source domain training data
    source_root = domain_roots.get(source_domain, f'{data_root}/{source_domain}')
    source_dataset = get_dataset(source_domain, source_root, 'train', train_transform)
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Target domain data
    target_root = domain_roots.get(target_domain, f'{data_root}/{target_domain}')

    # Few-shot loader (if n_shots > 0)
    fewshot_loader = None
    if n_shots > 0:
        target_train = get_dataset(target_domain, target_root, 'train', train_transform)

        # Random subset for few-shot
        indices = random.sample(range(len(target_train)), min(n_shots, len(target_train)))
        fewshot_dataset = Subset(target_train, indices)

        fewshot_loader = DataLoader(
            fewshot_dataset,
            batch_size=min(batch_size, n_shots),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Target test loader
    target_test = get_dataset(target_domain, target_root, 'test', test_transform)
    test_loader = DataLoader(
        target_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return source_loader, fewshot_loader, test_loader


class MultiDomainIterator:
    """
    Iterator that samples from multiple domain loaders

    Useful for joint training across domains
    """

    def __init__(self, loaders: Dict[str, DataLoader], sampling: str = 'uniform'):
        """
        Args:
            loaders: Dictionary of domain -> DataLoader
            sampling: 'uniform' (equal prob) or 'proportional' (by dataset size)
        """
        self.loaders = loaders
        self.sampling = sampling
        self.iterators = {k: iter(v) for k, v in loaders.items()}
        self.domains = list(loaders.keys())

        if not self.domains:
            raise ValueError(
                "No domain loaders were provided to MultiDomainIterator. "
                "Did 'create_multi_domain_loader' find any datasets? Check your data paths."
            )

        if sampling == 'proportional':
            sizes = [len(loaders[d].dataset) for d in self.domains]
            total = sum(sizes)
            if total == 0:
                raise ValueError("Found domain loaders but total dataset size is zero. Check datasets.")
            self.probs = [s / total for s in sizes]
        else:
            self.probs = [1.0 / len(self.domains)] * len(self.domains)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, dict]:
        """Return (domain_name, batch_dict)"""
        # Sample domain
        domain = random.choices(self.domains, weights=self.probs, k=1)[0]

        # Get batch, restart iterator if exhausted
        try:
            batch = next(self.iterators[domain])
        except StopIteration:
            self.iterators[domain] = iter(self.loaders[domain])
            batch = next(self.iterators[domain])

        batch['domain'] = domain
        return domain, batch

    def __len__(self):
        """Total number of batches across all domains"""
        return sum(len(loader) for loader in self.loaders.values())
