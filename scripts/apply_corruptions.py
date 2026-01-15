"""
Apply corruptions to clean GT images using imagecorruptions library
Creates paired (corrupted, clean) dataset for training
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from imagecorruptions import corrupt, get_corruption_names

# Available corruptions
CORRUPTION_CATEGORIES = {
    'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'weather': ['snow', 'frost', 'fog', 'brightness'],
    'digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'],
}

ALL_CORRUPTIONS = [c for cats in CORRUPTION_CATEGORIES.values() for c in cats]


def apply_corruptions(
    gt_dir: str,
    output_dir: str,
    corruptions: list = None,
    severities: list = [3],
    max_images: int = None,
):
    """
    Apply corruptions to clean images

    Args:
        gt_dir: Directory containing clean GT images
        output_dir: Output directory for corrupted images
        corruptions: List of corruption types (default: all)
        severities: List of severity levels 1-5 (default: [3])
        max_images: Maximum number of images to process (default: all)
    """
    gt_path = Path(gt_dir)
    output_path = Path(output_dir)

    if corruptions is None:
        corruptions = ALL_CORRUPTIONS

    # Get all JPEG files
    image_files = list(gt_path.rglob('*.JPEG')) + list(gt_path.rglob('*.jpeg')) + list(gt_path.rglob('*.jpg'))
    image_files = list(set(image_files))  # Remove duplicates if any

    if max_images:
        image_files = image_files[:max_images]

    print(f"Found {len(image_files)} images in {gt_dir}")
    print(f"Corruptions: {corruptions}")
    print(f"Severities: {severities}")
    print(f"Output: {output_dir}")

    for corruption in corruptions:
        for severity in severities:
            # Create output directory
            corrupt_dir = output_path / corruption / str(severity)
            corrupt_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nApplying {corruption} (severity={severity})...")

            for img_path in tqdm(image_files, desc=f"{corruption}/{severity}"):
                try:
                    # Load image
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img)

                    # Apply corruption
                    corrupted = corrupt(img_array, corruption_name=corruption, severity=severity)

                    # Save corrupted image
                    out_path = corrupt_dir / img_path.name
                    Image.fromarray(corrupted).save(out_path, quality=95)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

    print(f"\nDone! Corrupted images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Apply corruptions to GT images')
    parser.add_argument('--gt_dir', type=str, 
                        help='Directory containing clean GT images')
    parser.add_argument('--output_dir', type=str, 
                        help='Output directory for corrupted images')
    parser.add_argument('--corruptions', type=str, nargs='+', default=None,
                        help='Corruption types to apply (default: all)')
    parser.add_argument('--category', type=str, choices=['noise', 'blur', 'weather', 'digital'],
                        help='Apply all corruptions in a category')
    parser.add_argument('--severities', type=int, nargs='+', default=[3],
                        help='Severity levels 1-5 (default: 3)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process')

    args = parser.parse_args()

    # Determine corruptions to apply
    corruptions = args.corruptions
    if args.category:
        corruptions = CORRUPTION_CATEGORIES[args.category]

    apply_corruptions(
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        corruptions=corruptions,
        severities=args.severities,
        max_images=args.max_images,
    )


if __name__ == '__main__':
    main()
