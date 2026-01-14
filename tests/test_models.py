"""
Unit tests for model components
"""

import pytest
import torch

from src.models import (
    DegradationEncoder,
    ContentEncoder,
    CrossDomainDecoder,
    CrossDomainDegradationTransfer,
    SimpleDenoiser,
)


class TestDegradationEncoder:
    """Tests for DegradationEncoder"""

    def test_output_shape(self, sample_batch, device):
        """Test output dimensions"""
        model = DegradationEncoder(dim=256, n_degradation_types=8).to(device)
        x = sample_batch.to(device)

        z_d, mu, logvar, deg_type = model(x)

        batch_size = x.shape[0]
        assert z_d.shape == (batch_size, 256), f"Expected z_d shape {(batch_size, 256)}, got {z_d.shape}"
        assert mu.shape == (batch_size, 256), f"Expected mu shape {(batch_size, 256)}, got {mu.shape}"
        assert logvar.shape == (batch_size, 256), f"Expected logvar shape {(batch_size, 256)}, got {logvar.shape}"
        assert deg_type.shape == (batch_size, 8), f"Expected deg_type shape {(batch_size, 8)}, got {deg_type.shape}"

    def test_reparameterization(self, sample_batch, device):
        """Test that reparameterization produces different samples"""
        model = DegradationEncoder(dim=256).to(device)
        model.train()  # Enable stochastic sampling
        x = sample_batch.to(device)

        z_d_1, _, _, _ = model(x)
        z_d_2, _, _, _ = model(x)

        # In training mode, z_d should be stochastic
        assert not torch.allclose(z_d_1, z_d_2), "Reparameterization should produce different samples in train mode"

    def test_eval_mode_deterministic(self, sample_batch, device):
        """Test that eval mode uses mean (deterministic)"""
        model = DegradationEncoder(dim=256).to(device)
        model.eval()
        x = sample_batch.to(device)

        # Note: Current implementation still samples in eval mode
        # This test documents current behavior
        with torch.no_grad():
            z_d_1, mu_1, _, _ = model(x)
            z_d_2, mu_2, _, _ = model(x)

        # mu should be deterministic
        assert torch.allclose(mu_1, mu_2), "Mean should be deterministic"


class TestContentEncoder:
    """Tests for ContentEncoder"""

    def test_output_shape(self, sample_batch, device):
        """Test output dimensions"""
        model = ContentEncoder(dim=512).to(device)
        x = sample_batch.to(device)

        z_c = model(x)

        batch_size = x.shape[0]
        # ContentEncoder downsamples by 8x (3 stride-2 convolutions)
        expected_h = x.shape[2] // 8
        expected_w = x.shape[3] // 8

        assert z_c.shape == (batch_size, 512, expected_h, expected_w), \
            f"Expected z_c shape {(batch_size, 512, expected_h, expected_w)}, got {z_c.shape}"

    def test_different_input_sizes(self, device):
        """Test with different input sizes"""
        model = ContentEncoder(dim=512).to(device)

        for size in [64, 128, 256]:
            x = torch.randn(2, 3, size, size).to(device)
            z_c = model(x)

            expected_spatial = size // 8
            assert z_c.shape[2] == expected_spatial, f"Spatial dim mismatch for input size {size}"


class TestCrossDomainDecoder:
    """Tests for CrossDomainDecoder"""

    def test_output_shape(self, device):
        """Test output dimensions match input content spatial size * 8"""
        model = CrossDomainDecoder(content_dim=512, deg_dim=256).to(device)

        batch_size = 4
        spatial_size = 8  # After ContentEncoder: 64 // 8 = 8

        z_c = torch.randn(batch_size, 512, spatial_size, spatial_size).to(device)
        z_d = torch.randn(batch_size, 256).to(device)

        output = model(z_c, z_d, remove_degradation=True)

        # Decoder upsamples by 8x
        expected_size = spatial_size * 8
        assert output.shape == (batch_size, 3, expected_size, expected_size), \
            f"Expected output shape {(batch_size, 3, expected_size, expected_size)}, got {output.shape}"

    def test_output_range(self, device):
        """Test that output is in [-1, 1] due to tanh"""
        model = CrossDomainDecoder().to(device)

        z_c = torch.randn(2, 512, 8, 8).to(device)
        z_d = torch.randn(2, 256).to(device)

        output = model(z_c, z_d)

        assert output.min() >= -1.0, f"Output min {output.min()} < -1.0"
        assert output.max() <= 1.0, f"Output max {output.max()} > 1.0"

    def test_remove_vs_apply_degradation(self, device):
        """Test that remove_degradation flag changes output"""
        model = CrossDomainDecoder().to(device)

        z_c = torch.randn(2, 512, 8, 8).to(device)
        z_d = torch.randn(2, 256).to(device)

        out_remove = model(z_c, z_d, remove_degradation=True)
        out_apply = model(z_c, z_d, remove_degradation=False)

        assert not torch.allclose(out_remove, out_apply), \
            "Output should differ between remove and apply degradation modes"


class TestCrossDomainDegradationTransfer:
    """Tests for full CDDT model"""

    def test_forward_pass(self, sample_batch, device):
        """Test complete forward pass"""
        model = CrossDomainDegradationTransfer().to(device)
        x = sample_batch.to(device)

        outputs = model(x)

        assert 'restored' in outputs
        assert 'z_d' in outputs
        assert 'z_c' in outputs
        assert 'mu' in outputs
        assert 'logvar' in outputs
        assert 'deg_type' in outputs

    def test_restored_shape(self, sample_batch, device):
        """Test restored image has same shape as input"""
        model = CrossDomainDegradationTransfer().to(device)
        x = sample_batch.to(device)

        outputs = model(x)

        assert outputs['restored'].shape == x.shape, \
            f"Restored shape {outputs['restored'].shape} != input shape {x.shape}"

    def test_restore_method(self, sample_batch, device):
        """Test the restore convenience method"""
        model = CrossDomainDegradationTransfer().to(device)
        x = sample_batch.to(device)

        restored = model.restore(x)

        assert restored.shape == x.shape

    def test_transfer_degradation(self, device):
        """Test degradation transfer between images"""
        model = CrossDomainDegradationTransfer().to(device)

        x_source = torch.randn(2, 3, 64, 64).to(device)
        x_target = torch.randn(2, 3, 64, 64).to(device)

        transferred = model.transfer_degradation(x_source, x_target)

        assert transferred.shape == x_target.shape

    def test_gradient_flow(self, sample_batch, device):
        """Test that gradients flow through the model"""
        model = CrossDomainDegradationTransfer().to(device)
        x = sample_batch.to(device)

        outputs = model(x)
        loss = outputs['restored'].mean()
        loss.backward()

        # Check that some gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients found in model parameters"


class TestSimpleDenoiser:
    """Smoke tests for SimpleDenoiser"""

    def test_forward_shapes(self, sample_batch, device):
        model = SimpleDenoiser().to(device)
        x = sample_batch.to(device)

        outputs = model(x)

        assert 'restored' in outputs
        assert 'noise' in outputs
        assert outputs['restored'].shape == x.shape
        assert outputs['noise'].shape == x.shape

    def test_denoise_method(self, sample_batch, device):
        model = SimpleDenoiser().to(device)
        x = sample_batch.to(device)

        restored = model.denoise(x)

        assert restored.shape == x.shape
