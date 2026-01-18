# scripts/run_fewshot_transfer.py
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CrossDomainDegradationTransfer
from src.losses import ICMLLoss
from src.data import create_cross_domain_pairs
from src.utils import MetricTracker, compute_psnr, compute_ssim

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tracker = MetricTracker(["psnr", "ssim"])
    for batch in loader:
        degraded = batch["degraded"].to(device)
        clean = batch["clean"].to(device)
        out = model(degraded)
        restored = out["restored"] if isinstance(out, dict) else out
        restored = (restored + 1) / 2
        target = (clean + 1) / 2
        tracker.update("psnr", compute_psnr(restored, target))
        tracker.update("ssim", compute_ssim(restored, target))
    return tracker.compute()

def train_steps(model, loader, criterion, optimizer, device, steps: int, tag: str, writer=None, global_step_start=0):
    model.train()
    tracker = MetricTracker(["loss", "psnr"])
    it = iter(loader)
    pbar = tqdm(range(steps), desc=tag)

    for s in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        degraded = batch["degraded"].to(device)
        clean = batch["clean"].to(device)

        out = model(degraded)
        losses = criterion(out, clean)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        with torch.no_grad():
            restored = out["restored"] if isinstance(out, dict) else out
            restored = (restored + 1) / 2
            target = (clean + 1) / 2
            psnr = compute_psnr(restored, target)

        tracker.update("loss", losses["total"])
        tracker.update("psnr", psnr)
        pbar.set_postfix(loss=f"{losses['total'].item():.4f}", psnr=f"{psnr.item():.2f}")

        if writer and (s % 50 == 0):
            gs = global_step_start + s
            writer.add_scalar(f"{tag}/loss", losses["total"].item(), gs)
            writer.add_scalar(f"{tag}/psnr", psnr.item(), gs)

    return tracker.compute()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--source", type=str, default="imagenet-noise")
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--pretrain_epochs", type=int, default=10)
    ap.add_argument("--pretrain_iters", type=int, default=1000)
    ap.add_argument("--ft_steps", type=int, default=300)          # few-shot fine-tune steps
    ap.add_argument("--ft_lr", type=float, default=5e-5)          # fine-tune LR (작게)
    args = ap.parse_args()

    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Experiment dir
    exp_name = f"transfer_{args.source}_to_{args.target}_shots{args.shots}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(cfg["experiment"]["save_dir"]) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    writer = SummaryWriter(exp_dir / "logs")

    # Model + loss
    model = CrossDomainDegradationTransfer(
        deg_dim=cfg["model"]["deg_dim"],
        content_dim=cfg["model"]["content_dim"],
        n_degradation_types=cfg["model"]["n_degradation_types"],
    ).to(device)

    criterion = ICMLLoss(
        recon_weight=cfg["loss"]["recon_weight"],
        kl_weight=cfg["loss"]["kl_weight"],
        disentangle_weight=cfg["loss"]["disentangle_weight"],
        domain_weight=cfg["loss"]["domain_weight"],
    )

    # Loaders
    source_loader, fewshot_loader, test_loader = create_cross_domain_pairs(
        source_domain=args.source,
        target_domain=args.target,
        data_root=cfg["data"]["train_dir"].replace("/train", ""),
        n_shots=args.shots,
        batch_size=cfg["training"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    # 1) Pretrain on source
    opt = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    global_step = 0
    for ep in range(args.pretrain_epochs):
        # pretrain: source only
        train_steps(model, source_loader, criterion, opt, device,
                    steps=args.pretrain_iters, tag="pretrain", writer=writer, global_step_start=global_step)
        global_step += args.pretrain_iters

        # zero-shot eval on target
        res = evaluate(model, test_loader, device)
        writer.add_scalar(f"val/{args.target}_psnr", res["psnr"], ep)
        writer.add_scalar(f"val/{args.target}_ssim", res["ssim"], ep)
        print(f"[Pretrain ep {ep}] target {args.target}: {res}")

    # Save pretrain ckpt
    torch.save({"model": model.state_dict()}, exp_dir / "pretrain.pth")

    # 2) Few-shot fine-tune (if shots > 0)
    if args.shots > 0:
        ft_opt = optim.Adam(model.parameters(), lr=args.ft_lr)
        train_steps(model, fewshot_loader, criterion, ft_opt, device,
                    steps=args.ft_steps, tag="fewshot_ft", writer=writer, global_step_start=global_step)
        global_step += args.ft_steps

    # 3) Final test on target
    final = evaluate(model, test_loader, device)
    print(f"[Final] target {args.target} shots={args.shots}: {final}")
    writer.add_scalar(f"final/{args.target}_psnr", final["psnr"], 0)
    writer.add_scalar(f"final/{args.target}_ssim", final["ssim"], 0)

    torch.save({"model": model.state_dict()}, exp_dir / "final.pth")
    writer.close()
    print(f"Saved to: {exp_dir}")

if __name__ == "__main__":
    main()
