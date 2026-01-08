"""
Dataset classes for multi-domain image restoration

Supported domains:
- ImageNet-C: Natural images with synthetic corruptions
- LDCT: Low-dose CT medical images
- DIBCO: Document image binarization
- FMD: Fluorescence microscopy denoising
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class PairedImageDataset(Dataset):
    """Base class for paired degraded-clean image datasets"""

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        degraded_folder: str = 'degraded',
        clean_folder: str = 'clean',
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.degraded_folder = degraded_folder
        self.clean_folder = clean_folder

        self.degraded_dir = self.root / split / degraded_folder
        self.clean_dir = self.root / split / clean_folder

        self.image_pairs = self._load_image_pairs()

    def _load_image_pairs(self) -> List[Tuple[Path, Path]]:
        """Load paired image paths"""
        pairs = []

        if not self.degraded_dir.exists():
            return pairs

        for degraded_path in sorted(self.degraded_dir.glob('*')):
            if degraded_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                clean_path = self.clean_dir / degraded_path.name
                if clean_path.exists():
                    pairs.append((degraded_path, clean_path))

        return pairs

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> dict:
        degraded_path, clean_path = self.image_pairs[idx]

        degraded = Image.open(degraded_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')

        if self.transform:
            degraded = self.transform(degraded)
            clean = self.transform(clean)

        return {
            'degraded': degraded,
            'clean': clean,
            'path': str(degraded_path),
        }


class ImageNetCDataset(PairedImageDataset):
    """
    ImageNet-C: Natural images with synthetic corruptions

    Corruptions: noise, blur, weather, digital
    Severity levels: 1-5

    Download: https://github.com/hendrycks/robustness
    """

    CORRUPTION_TYPES = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',  # Noise
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',  # Blur
        'snow', 'frost', 'fog', 'brightness',  # Weather
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',  # Digital
    ]

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        corruption_type: Optional[str] = None,
        severity: int = 3,
    ):
        self.corruption_type = corruption_type
        self.severity = severity
        super().__init__(root, split, transform)

    def _load_image_pairs(self) -> List[Tuple[Path, Path]]:
        """Load ImageNet-C pairs based on corruption type and severity"""
        pairs = []
        root = Path(self.root)

        corruptions = [self.corruption_type] if self.corruption_type else self.CORRUPTION_TYPES

        for corruption in corruptions:
            degraded_dir = root / corruption / str(self.severity)
            clean_dir = root / 'clean'

            if not degraded_dir.exists():
                continue

            for degraded_path in sorted(degraded_dir.glob('**/*')):
                if degraded_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    # ImageNet-C structure: corruption/severity/class/image.jpg
                    relative_path = degraded_path.relative_to(degraded_dir)
                    clean_path = clean_dir / relative_path

                    if clean_path.exists():
                        pairs.append((degraded_path, clean_path))

        return pairs


class LDCTDataset(PairedImageDataset):
    """
    LDCT Grand Challenge: Low-dose CT denoising

    Paired low-dose and normal-dose CT images

    Download: https://www.aapm.org/grandchallenge/lowdosect/
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            root, split, transform,
            degraded_folder='low_dose',
            clean_folder='full_dose'
        )

    def __getitem__(self, idx: int) -> dict:
        degraded_path, clean_path = self.image_pairs[idx]

        # CT images are typically 16-bit grayscale
        degraded = self._load_ct_image(degraded_path)
        clean = self._load_ct_image(clean_path)

        if self.transform:
            degraded = self.transform(degraded)
            clean = self.transform(clean)

        return {
            'degraded': degraded,
            'clean': clean,
            'path': str(degraded_path),
            'domain': 'ldct',
        }

    def _load_ct_image(self, path: Path) -> Image.Image:
        """Load CT image and normalize to 8-bit for model input"""
        img = Image.open(path)

        # Convert to numpy for processing
        arr = np.array(img, dtype=np.float32)

        # Normalize to [0, 255]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        arr = arr.astype(np.uint8)

        # Convert to RGB (replicate grayscale)
        return Image.fromarray(arr).convert('RGB')


class DIBCODataset(PairedImageDataset):
    """
    DIBCO: Document Image Binarization Competition

    Degraded historical document images with ground truth

    Download: https://vc.ee.duth.gr/dibco/
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        year: int = 2019,
    ):
        self.year = year
        super().__init__(
            root, split, transform,
            degraded_folder='imgs',
            clean_folder='gt'
        )

    def _load_image_pairs(self) -> List[Tuple[Path, Path]]:
        """Load DIBCO pairs"""
        pairs = []
        root = Path(self.root) / str(self.year) / self.split

        degraded_dir = root / 'imgs'
        clean_dir = root / 'gt'

        if not degraded_dir.exists():
            return pairs

        for degraded_path in sorted(degraded_dir.glob('*')):
            if degraded_path.suffix.lower() in ['.png', '.jpg', '.bmp', '.tif']:
                # GT files may have different extension
                stem = degraded_path.stem
                for ext in ['.png', '.bmp', '.tif']:
                    clean_path = clean_dir / (stem + ext)
                    if clean_path.exists():
                        pairs.append((degraded_path, clean_path))
                        break

        return pairs


class FMDDataset(PairedImageDataset):
    """
    FMD: Fluorescence Microscopy Denoising Dataset

    Noisy and averaged (clean) fluorescence microscopy images

    Download: https://github.com/yinhaoz/denoising-fluorescence
    """

    MICROSCOPY_TYPES = [
        'Confocal_BPAE',
        'Confocal_MICE',
        'TwoPhoton_BPAE',
        'TwoPhoton_MICE',
        'WideField_BPAE',
    ]

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        microscopy_type: Optional[str] = None,
    ):
        self.microscopy_type = microscopy_type
        super().__init__(root, split, transform)

    def _load_image_pairs(self) -> List[Tuple[Path, Path]]:
        """Load FMD pairs"""
        pairs = []
        root = Path(self.root)

        types = [self.microscopy_type] if self.microscopy_type else self.MICROSCOPY_TYPES

        for mtype in types:
            type_dir = root / mtype / self.split

            if not type_dir.exists():
                continue

            # FMD structure: type/split/noisy/ and type/split/gt/
            noisy_dir = type_dir / 'noisy'
            gt_dir = type_dir / 'gt'

            if not noisy_dir.exists():
                continue

            for noisy_path in sorted(noisy_dir.glob('*.png')):
                gt_path = gt_dir / noisy_path.name
                if gt_path.exists():
                    pairs.append((noisy_path, gt_path))

        return pairs


def get_dataset(domain: str, root: str, split: str = 'train', transform=None, **kwargs):
    """Factory function to create dataset by domain name"""

    datasets = {
        'imagenet': ImageNetCDataset,
        'imagenet-c': ImageNetCDataset,
        'ldct': LDCTDataset,
        'dibco': DIBCODataset,
        'fmd': FMDDataset,
        'microscopy': FMDDataset,
    }

    domain_lower = domain.lower()
    if domain_lower not in datasets:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(datasets.keys())}")

    return datasets[domain_lower](root, split, transform, **kwargs)
