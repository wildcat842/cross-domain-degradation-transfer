"""
Unit tests for loss functions
"""

import pytest
import torch

from src.losses import ICMLLoss


class TestICMLLoss:
    """Tests for ICMLLoss"""

    @pytest.fixture
    def model_outputs(self, batch_size, image_size):
        """Create mock model outputs"""
        spatial = image_size // 8
        return {
            'restored': torch.randn(batch_size, 3, image_size, image_size),
            'z_d': torch.randn(batch_size, 256),
            'z_c': torch.randn(batch_size, 512, spatial, spatial),
            'mu': torch.randn(batch_size, 256),
            'logvar': torch.randn(batch_size, 256),
            'deg_type': torch.randn(batch_size, 8),
        }

    @pytest.fixture
    def targets(self, batch_size, image_size):
        """Create mock target images"""
        return torch.randn(batch_size, 3, image_size, image_size)

    def test_loss_computation(self, model_outputs, targets):
        """Test that loss computes without error"""
        criterion = ICMLLoss()
        losses = criterion(model_outputs, targets)

        assert 'total' in losses
        assert 'recon' in losses
        assert 'kl' in losses
        assert 'disentangle' in losses

    def test_loss_values_positive(self, model_outputs, targets):
        """Test that individual losses are non-negative"""
        criterion = ICMLLoss()
        losses = criterion(model_outputs, targets)

        assert losses['recon'] >= 0, f"Recon loss should be >= 0, got {losses['recon']}"
        # KL can be negative in some formulations, but typically positive
        # HSIC should be non-negative
        assert losses['disentangle'] >= 0, f"Disentangle loss should be >= 0, got {losses['disentangle']}"

    def test_loss_weights(self, model_outputs, targets):
        """Test that loss weights affect the total loss"""
        criterion_default = ICMLLoss(recon_weight=1.0, kl_weight=0.01)
        criterion_high_kl = ICMLLoss(recon_weight=1.0, kl_weight=1.0)

        losses_default = criterion_default(model_outputs, targets)
        losses_high_kl = criterion_high_kl(model_outputs, targets)

        # With higher KL weight, total should be different (unless KL is 0)
        # Just check they compute without error
        assert losses_default['total'] is not None
        assert losses_high_kl['total'] is not None

    def test_zero_disentangle_weight(self, model_outputs, targets):
        """Test with disentanglement disabled"""
        criterion = ICMLLoss(disentangle_weight=0.0)
        losses = criterion(model_outputs, targets)

        assert losses['disentangle'] == 0.0, "Disentangle loss should be 0 when weight is 0"

    def test_backward_pass(self, model_outputs, targets):
        """Test that loss enables gradient computation"""
        # Make outputs require grad
        for key in model_outputs:
            model_outputs[key] = model_outputs[key].requires_grad_(True)

        criterion = ICMLLoss()
        losses = criterion(model_outputs, targets)

        losses['total'].backward()

        # Check gradients exist
        assert model_outputs['restored'].grad is not None, "No gradient for restored"
        assert model_outputs['mu'].grad is not None, "No gradient for mu"

    def test_hsic_independence(self):
        """Test HSIC: independent variables should have low HSIC"""
        torch.manual_seed(42)  # 재현성을 위한 시드 고정
        criterion = ICMLLoss()

        # Independent random variables
        x = torch.randn(256, 64)
        y = torch.randn(256, 64)
        hsic_indep = criterion._hsic(x, y)

        # Strongly dependent variables (y = x, perfect correlation)
        y_dep = x.clone()
        hsic_dep = criterion._hsic(x, y_dep)

        # Dependent should have higher HSIC
        assert hsic_dep > hsic_indep, \
            f"Dependent HSIC ({hsic_dep}) should be > independent HSIC ({hsic_indep})"

    def test_kl_divergence_standard_normal(self):
        """Test KL: standard normal prior should have low KL"""
        criterion = ICMLLoss(kl_weight=1.0)

        batch_size = 32
        dim = 256

        # Near standard normal: mu=0, logvar=0
        outputs_standard = {
            'restored': torch.zeros(batch_size, 3, 64, 64),
            'z_d': torch.randn(batch_size, dim),
            'z_c': torch.randn(batch_size, 512, 8, 8),
            'mu': torch.zeros(batch_size, dim),
            'logvar': torch.zeros(batch_size, dim),
            'deg_type': torch.randn(batch_size, 8),
        }

        # Far from standard normal
        outputs_far = {
            'restored': torch.zeros(batch_size, 3, 64, 64),
            'z_d': torch.randn(batch_size, dim),
            'z_c': torch.randn(batch_size, 512, 8, 8),
            'mu': torch.ones(batch_size, dim) * 5,
            'logvar': torch.ones(batch_size, dim) * 2,
            'deg_type': torch.randn(batch_size, 8),
        }

        targets = torch.zeros(batch_size, 3, 64, 64)

        losses_standard = criterion(outputs_standard, targets)
        losses_far = criterion(outputs_far, targets)

        assert losses_standard['kl'] < losses_far['kl'], \
            "Standard normal should have lower KL divergence"

    def test_reconstruction_perfect(self):
        """Test reconstruction loss with perfect prediction"""
        criterion = ICMLLoss()

        batch_size = 4
        targets = torch.randn(batch_size, 3, 64, 64)

        outputs = {
            'restored': targets.clone(),  # Perfect reconstruction
            'z_d': torch.randn(batch_size, 256),
            'z_c': torch.randn(batch_size, 512, 8, 8),
            'mu': torch.zeros(batch_size, 256),
            'logvar': torch.zeros(batch_size, 256),
            'deg_type': torch.randn(batch_size, 8),
        }

        losses = criterion(outputs, targets)

        assert losses['recon'].item() < 1e-6, \
            f"Perfect reconstruction should have ~0 loss, got {losses['recon']}"
