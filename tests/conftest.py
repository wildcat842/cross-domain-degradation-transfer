"""
Pytest configuration and shared fixtures
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def device():
    """Get available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def batch_size():
    """Default batch size for tests"""
    return 4


@pytest.fixture
def image_size():
    """Default image size for tests"""
    return 64  # Smaller for faster tests


@pytest.fixture
def sample_batch(batch_size, image_size):
    """Create a sample batch of images"""
    return torch.randn(batch_size, 3, image_size, image_size)


@pytest.fixture
def sample_pair(batch_size, image_size):
    """Create a sample pair of degraded and clean images"""
    degraded = torch.randn(batch_size, 3, image_size, image_size)
    clean = torch.randn(batch_size, 3, image_size, image_size)
    return degraded, clean
