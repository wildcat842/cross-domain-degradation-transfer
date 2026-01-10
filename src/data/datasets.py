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


class ImageNetCDataset(Dataset):
    """
    ImageNet-C / Tiny-ImageNet-C: Natural images with synthetic corruptions

    Corruptions: noise, blur, weather, digital
    Severity levels: 1-5

    Supports two modes:
    1. Paired mode (with clean): {root}/{corruption}/{severity}/{class}/{image}
                                 {root}/clean/{class}/{image}
    2. Degraded-only mode (Tiny-ImageNet-C): {root}/Tiny-ImageNet-C/{corruption}/{severity}/{class}/{image}
       In this mode, uses different corruption as "target" for transfer learning

    Download: https://github.com/hendrycks/robustness
    """

    CORRUPTION_TYPES = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',  # Noise
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',  # Blur
        'snow', 'frost', 'fog', 'brightness',  # Weather
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',  # Digital
    ]

    # Tiny-ImageNet-C available corruptions (11 types)
    TINY_CORRUPTION_TYPES = [
        'shot_noise', 'impulse_noise',  # Noise
        'defocus_blur', 'glass_blur', 'motion_blur',  # Blur
        'frost', 'brightness',  # Weather
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
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.corruption_type = corruption_type
        self.severity = severity

        # Detect if this is Tiny-ImageNet-C structure
        self.is_tiny = (self.root / 'Tiny-ImageNet-C').exists()
        if self.is_tiny:
            self.base_dir = self.root / 'Tiny-ImageNet-C'
        else:
            self.base_dir = self.root

        self.images = self._load_images()

    def _load_images(self) -> List[Tuple[Path, Optional[Path]]]:
        """Load images (paired or degraded-only)"""
        images = []

        # Determine which corruptions to use
        available_corruptions = self.TINY_CORRUPTION_TYPES if self.is_tiny else self.CORRUPTION_TYPES
        corruptions = [self.corruption_type] if self.corruption_type else available_corruptions

        for corruption in corruptions:
            degraded_dir = self.base_dir / corruption / str(self.severity)

            if not degraded_dir.exists():
                continue

            for degraded_path in sorted(degraded_dir.glob('**/*')):
                if degraded_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    # Check for paired clean image
                    relative_path = degraded_path.relative_to(degraded_dir)

                    if self.is_tiny:
                        # Tiny-ImageNet-C: no clean, store None
                        images.append((degraded_path, None))
                    else:
                        clean_dir = self.base_dir / 'clean'
                        clean_path = clean_dir / relative_path
                        if clean_path.exists():
                            images.append((degraded_path, clean_path))

        return images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        degraded_path, clean_path = self.images[idx]

        degraded = Image.open(degraded_path).convert('RGB')

        if self.transform:
            degraded = self.transform(degraded)

        result = {
            'degraded': degraded,
            'path': str(degraded_path),
            'domain': 'imagenet-c',
        }

        if clean_path is not None:
            clean = Image.open(clean_path).convert('RGB')
            if self.transform:
                clean = self.transform(clean)
            result['clean'] = clean
        else:
            # For degraded-only mode, use degraded as target (self-supervised)
            result['clean'] = degraded

        return result


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

        # 두 가지 경로 패턴 지원:
        # 1. {root}/{year}/imgs/ (split 없음)
        # 2. {root}/{year}/{split}/imgs/ (split 있음)
        root_no_split = Path(self.root) / str(self.year)
        root_with_split = Path(self.root) / str(self.year) / self.split

        # split 없는 구조 먼저 시도
        if (root_no_split / 'imgs').exists():
            degraded_dir = root_no_split / 'imgs'
            clean_dir = root_no_split / 'gt'
        else:
            degraded_dir = root_with_split / 'imgs'
            clean_dir = root_with_split / 'gt'

        if not degraded_dir.exists():
            return pairs

        for degraded_path in sorted(degraded_dir.glob('*')):
            if degraded_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
                # GT files may have different extension
                stem = degraded_path.stem
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
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

    Flat 폴더 구조:
        noisy/{microscopy_type}_{capture_id}_{filename}.png
        clean/{microscopy_type}_{capture_id}.png
    """

    MICROSCOPY_TYPES = [
        'Confocal_BPAE_B',
        'Confocal_BPAE_G',
        'Confocal_BPAE_R',
        'Confocal_FISH',
        'Confocal_MICE',
        'TwoPhoton_BPAE_B',
        'TwoPhoton_BPAE_G',
        'TwoPhoton_BPAE_R',
        'TwoPhoton_MICE',
        'WideField_BPAE_B',
        'WideField_BPAE_G',
        'WideField_BPAE_R',
    ]

    def __init__(
        self,
        root: str,
        split: str = 'train',  # FMD는 split 구분 없음
        transform: Optional[Callable] = None,
        microscopy_type: Optional[str] = None,
    ):
        self.microscopy_type = microscopy_type
        self.root = Path(root)
        self.transform = transform
        self.image_pairs = self._load_image_pairs()

    def _load_image_pairs(self) -> List[Tuple[Path, Path]]:
        """Load FMD pairs from flat noisy/clean structure"""
        pairs = []

        noisy_dir = self.root / 'noisy'
        clean_dir = self.root / 'clean'

        if not noisy_dir.exists() or not clean_dir.exists():
            return pairs

        # clean 파일 목록으로 prefix 매핑 생성
        # clean 파일명: {microscopy_type}_{capture_id}.png
        clean_prefixes = {}
        for clean_path in clean_dir.glob('*.png'):
            prefix = clean_path.stem  # e.g., Confocal_BPAE_B_1
            clean_prefixes[prefix] = clean_path

        # noisy 이미지 탐색
        for noisy_path in sorted(noisy_dir.glob('*.png')):
            # 파일명: {microscopy_type}_{capture_id}_{rest}.png
            # e.g., Confocal_BPAE_B_1_HV110_P0500510000.png
            noisy_stem = noisy_path.stem

            # clean prefix와 매칭 찾기
            matched_prefix = None
            for prefix in clean_prefixes:
                if noisy_stem.startswith(prefix + '_'):
                    matched_prefix = prefix
                    break

            if matched_prefix is None:
                continue

            clean_path = clean_prefixes[matched_prefix]

            # microscopy_type 필터링
            if self.microscopy_type:
                if not matched_prefix.startswith(self.microscopy_type):
                    continue

            pairs.append((noisy_path, clean_path))

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
            'domain': 'fmd',
        }


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
