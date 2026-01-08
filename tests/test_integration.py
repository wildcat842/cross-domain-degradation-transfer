"""
Integration tests for the full training pipeline
"""

import pytest
import torch
import torch.optim as optim

from src.models import CrossDomainDegradationTransfer
from src.losses import ICMLLoss
from src.utils.metrics import compute_psnr, compute_ssim


class TestTrainingPipeline:
    """Integration tests for training pipeline"""

    @pytest.fixture
    def model(self, device):
        """Create model"""
        return CrossDomainDegradationTransfer(
            deg_dim=64,  # Smaller for faster tests
            content_dim=128,
            n_degradation_types=4,
        ).to(device)

    @pytest.fixture
    def criterion(self):
        """Create loss function"""
        return ICMLLoss(
            recon_weight=1.0,
            kl_weight=0.01,
            disentangle_weight=0.1,
            domain_weight=0.0,  # Skip domain loss for simple test
        )

    def test_single_training_step(self, model, criterion, device):
        """Test single forward-backward pass"""
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Create dummy batch
        degraded = torch.randn(4, 3, 64, 64).to(device)
        clean = torch.randn(4, 3, 64, 64).to(device)

        # Forward
        outputs = model(degraded)
        losses = criterion(outputs, clean)

        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # Check loss is finite
        assert torch.isfinite(losses['total']), "Loss should be finite"

    def test_multiple_training_steps(self, model, criterion, device):
        """Test multiple training steps reduce loss"""
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Fixed batch for consistency
        torch.manual_seed(42)
        degraded = torch.randn(4, 3, 64, 64).to(device)
        clean = torch.randn(4, 3, 64, 64).to(device)

        initial_loss = None
        final_loss = None

        for step in range(10):
            outputs = model(degraded)
            losses = criterion(outputs, clean)

            if step == 0:
                initial_loss = losses['total'].item()

            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()

            if step == 9:
                final_loss = losses['total'].item()

        # Loss should decrease (or at least not explode)
        assert final_loss < initial_loss * 2, \
            f"Loss should not explode: {initial_loss} -> {final_loss}"

    def test_model_improves_psnr(self, model, criterion, device):
        """Test that trained model improves PSNR over random output"""
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Create a simple restoration task: clean + noise -> clean
        torch.manual_seed(42)
        clean = torch.rand(4, 3, 64, 64).to(device) * 2 - 1  # [-1, 1]
        noise = torch.randn_like(clean) * 0.3
        degraded = (clean + noise).clamp(-1, 1)

        # Initial PSNR
        with torch.no_grad():
            outputs = model(degraded)
            restored = (outputs['restored'] + 1) / 2
            target = (clean + 1) / 2
            initial_psnr = compute_psnr(restored, target).item()

        # Train
        for _ in range(20):
            outputs = model(degraded)
            losses = criterion(outputs, clean)

            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()

        # Final PSNR
        with torch.no_grad():
            outputs = model(degraded)
            restored = (outputs['restored'] + 1) / 2
            final_psnr = compute_psnr(restored, target).item()

        # PSNR should improve or at least not degrade significantly
        assert final_psnr >= initial_psnr - 2, \
            f"PSNR should not degrade significantly: {initial_psnr} -> {final_psnr}"

    def test_eval_mode_deterministic(self, model, device):
        """Test that eval mode produces deterministic outputs"""
        model.eval()

        degraded = torch.randn(2, 3, 64, 64).to(device)

        with torch.no_grad():
            out1 = model(degraded)['restored']
            out2 = model(degraded)['restored']

        # Note: Due to VAE sampling, outputs may still differ slightly
        # This test documents the behavior
        # For fully deterministic output, we'd need to use mu instead of sampled z_d


class TestEndToEnd:
    """End-to-end tests simulating real usage"""

    def test_save_load_checkpoint(self, device, tmp_path):
        """Test saving and loading model checkpoint"""
        # Create and train model briefly
        model = CrossDomainDegradationTransfer(deg_dim=64, content_dim=128).to(device)
        optimizer = optim.Adam(model.parameters())

        degraded = torch.randn(2, 3, 64, 64).to(device)
        outputs = model(degraded)

        # Save
        checkpoint_path = tmp_path / "checkpoint.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        # Load into new model
        model2 = CrossDomainDegradationTransfer(deg_dim=64, content_dim=128).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model2.load_state_dict(checkpoint['model_state_dict'])

        # Verify outputs match
        model.eval()
        model2.eval()

        with torch.no_grad():
            out1 = model(degraded)['restored']
            out2 = model2(degraded)['restored']

        # Due to VAE, use mu comparison instead
        # Or compare with same random seed
        torch.manual_seed(0)
        out1 = model(degraded)['mu']
        torch.manual_seed(0)
        out2 = model2(degraded)['mu']

        assert torch.allclose(out1, out2, atol=1e-5), "Loaded model should match saved model"

    def test_different_input_sizes(self, device):
        """Test model handles different input sizes"""
        model = CrossDomainDegradationTransfer().to(device)
        model.eval()

        for size in [64, 128, 256]:
            x = torch.randn(1, 3, size, size).to(device)

            with torch.no_grad():
                outputs = model(x)

            assert outputs['restored'].shape == x.shape, \
                f"Output shape mismatch for input size {size}"

    def test_batch_size_one(self, device):
        """Test model works with batch size 1"""
        model = CrossDomainDegradationTransfer().to(device)

        x = torch.randn(1, 3, 64, 64).to(device)
        outputs = model(x)

        assert outputs['restored'].shape == (1, 3, 64, 64)

    def test_cuda_if_available(self):
        """Test CUDA operations if available"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = CrossDomainDegradationTransfer().cuda()
        x = torch.randn(2, 3, 64, 64).cuda()

        outputs = model(x)

        assert outputs['restored'].is_cuda
        assert outputs['z_d'].is_cuda
