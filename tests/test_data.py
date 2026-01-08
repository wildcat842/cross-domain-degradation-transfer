"""
Unit tests for data loading utilities
"""

import pytest
import torch
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from src.data.datasets import (
    PairedImageDataset,
    ImageNetCDataset,
    LDCTDataset,
    DIBCODataset,
    FMDDataset,
    get_dataset,
)
from src.data.loader import (
    get_default_transform,
    MultiDomainIterator,
)


class TestTransforms:
    """Tests for image transforms"""

    def test_train_transform_shape(self):
        """Test training transform output shape"""
        transform = get_default_transform(image_size=256, train=True)

        # Create a dummy PIL image
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        transformed = transform(img)

        assert transformed.shape == (3, 256, 256), \
            f"Expected shape (3, 256, 256), got {transformed.shape}"

    def test_test_transform_shape(self):
        """Test test transform output shape"""
        transform = get_default_transform(image_size=256, train=False)

        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        transformed = transform(img)

        assert transformed.shape == (3, 256, 256)

    def test_transform_normalization(self):
        """Test that transform normalizes to [-1, 1]"""
        transform = get_default_transform(image_size=64, train=False)

        # White image
        img = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 255)
        transformed = transform(img)

        # After normalization with mean=0.5, std=0.5: (1.0 - 0.5) / 0.5 = 1.0
        assert transformed.max() <= 1.0 + 1e-5
        assert transformed.min() >= -1.0 - 1e-5


class TestPairedImageDataset:
    """Tests for PairedImageDataset base class"""

    @pytest.fixture
    def temp_dataset(self):
        """Create temporary dataset directory with sample images"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create directory structure
            (tmpdir / 'train' / 'degraded').mkdir(parents=True)
            (tmpdir / 'train' / 'clean').mkdir(parents=True)

            # Create sample images
            for i in range(5):
                img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                img.save(tmpdir / 'train' / 'degraded' / f'img_{i}.png')
                img.save(tmpdir / 'train' / 'clean' / f'img_{i}.png')

            yield tmpdir

    def test_dataset_length(self, temp_dataset):
        """Test dataset returns correct length"""
        dataset = PairedImageDataset(str(temp_dataset), split='train')
        assert len(dataset) == 5

    def test_dataset_item(self, temp_dataset):
        """Test dataset returns correct item format"""
        transform = get_default_transform(image_size=64, train=False)
        dataset = PairedImageDataset(str(temp_dataset), split='train', transform=transform)

        item = dataset[0]

        assert 'degraded' in item
        assert 'clean' in item
        assert 'path' in item
        assert item['degraded'].shape == (3, 64, 64)
        assert item['clean'].shape == (3, 64, 64)

    def test_empty_dataset(self):
        """Test dataset handles missing directory gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = PairedImageDataset(tmpdir, split='train')
            assert len(dataset) == 0


class TestGetDataset:
    """Tests for dataset factory function"""

    def test_unknown_domain_raises(self):
        """Test that unknown domain raises ValueError"""
        with pytest.raises(ValueError, match="Unknown domain"):
            get_dataset('unknown_domain', '/fake/path')

    def test_valid_domain_names(self):
        """Test that valid domain names are accepted"""
        valid_domains = ['imagenet', 'imagenet-c', 'ldct', 'dibco', 'fmd', 'microscopy']

        for domain in valid_domains:
            # Should not raise, even if directory doesn't exist
            try:
                dataset = get_dataset(domain, '/fake/path')
                assert len(dataset) == 0  # No files, but no error
            except Exception as e:
                pytest.fail(f"get_dataset raised {e} for valid domain {domain}")


class TestMultiDomainIterator:
    """Tests for MultiDomainIterator"""

    @pytest.fixture
    def mock_loaders(self):
        """Create mock data loaders"""
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, size):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    'degraded': torch.randn(3, 64, 64),
                    'clean': torch.randn(3, 64, 64),
                }

        loaders = {
            'domain_a': torch.utils.data.DataLoader(MockDataset(10), batch_size=2),
            'domain_b': torch.utils.data.DataLoader(MockDataset(20), batch_size=2),
        }
        return loaders

    def test_iterator_returns_domain(self, mock_loaders):
        """Test that iterator returns domain name with batch"""
        iterator = MultiDomainIterator(mock_loaders)

        domain, batch = next(iterator)

        assert domain in ['domain_a', 'domain_b']
        assert 'degraded' in batch
        assert 'domain' in batch

    def test_iterator_cycles(self, mock_loaders):
        """Test that iterator cycles through data"""
        iterator = MultiDomainIterator(mock_loaders)

        # Should be able to iterate more times than total batches
        total_batches = len(mock_loaders['domain_a']) + len(mock_loaders['domain_b'])

        for i in range(total_batches + 5):
            domain, batch = next(iterator)
            assert batch['degraded'] is not None

    def test_uniform_sampling(self, mock_loaders):
        """Test uniform sampling across domains"""
        iterator = MultiDomainIterator(mock_loaders, sampling='uniform')

        domain_counts = {'domain_a': 0, 'domain_b': 0}

        for _ in range(100):
            domain, _ = next(iterator)
            domain_counts[domain] += 1

        # With uniform sampling, counts should be roughly equal
        ratio = domain_counts['domain_a'] / domain_counts['domain_b']
        assert 0.5 < ratio < 2.0, f"Uniform sampling should be balanced, got ratio {ratio}"


class TestMetrics:
    """Tests for evaluation metrics"""

    def test_psnr_identical(self):
        """Test PSNR with identical images"""
        from src.utils.metrics import compute_psnr

        img = torch.rand(2, 3, 64, 64)
        psnr = compute_psnr(img, img)

        # PSNR of identical images should be very high (technically infinite)
        assert psnr > 50, f"PSNR of identical images should be high, got {psnr}"

    def test_psnr_different(self):
        """Test PSNR with different images"""
        from src.utils.metrics import compute_psnr

        img1 = torch.rand(2, 3, 64, 64)
        img2 = torch.rand(2, 3, 64, 64)
        psnr = compute_psnr(img1, img2)

        # PSNR of random images should be low
        assert 0 < psnr < 50, f"PSNR of random images should be moderate, got {psnr}"

    def test_ssim_identical(self):
        """Test SSIM with identical images"""
        from src.utils.metrics import compute_ssim

        img = torch.rand(2, 3, 64, 64)
        ssim = compute_ssim(img, img)

        # SSIM of identical images should be 1.0
        assert ssim > 0.99, f"SSIM of identical images should be ~1.0, got {ssim}"

    def test_ssim_range(self):
        """Test SSIM is in valid range"""
        from src.utils.metrics import compute_ssim

        img1 = torch.rand(2, 3, 64, 64)
        img2 = torch.rand(2, 3, 64, 64)
        ssim = compute_ssim(img1, img2)

        assert -1 <= ssim <= 1, f"SSIM should be in [-1, 1], got {ssim}"

    def test_metric_tracker(self):
        """Test MetricTracker aggregation"""
        from src.utils.metrics import MetricTracker

        tracker = MetricTracker(['psnr', 'ssim'])

        tracker.update('psnr', 30.0)
        tracker.update('psnr', 32.0)
        tracker.update('ssim', 0.9)
        tracker.update('ssim', 0.95)

        results = tracker.compute()

        assert results['psnr'] == 31.0  # (30 + 32) / 2
        assert results['ssim'] == 0.925  # (0.9 + 0.95) / 2
