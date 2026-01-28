"""
Few-shot adaptation for Cross-Domain / Cross-Corruption Transfer

- Load source-specific pretrained checkpoint
- Zero-shot evaluation on target
- Few-shot fine-tuning (optional)



python finetune_fewshot.py --fewshot.yaml --source <source_domain> --target <target_domain> --shots <n_shots> --ft_steps <fine_tune_steps> --ft_lr <fine_tune_lr>
python ./scripts/finetune_fewshot.py --config configs/fewshot.yaml --source imagenet-blur --target imagenet-noise --shots 100  

"""

import argparse
from pathlib import Path
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import sys
import csv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import CrossDomainDegradationTransfer
from src.losses import ICMLLoss
from src.data import create_cross_domain_pairs
from src.utils import MetricTracker, compute_psnr, compute_ssim


# -------------------------------------------------
# Utils
# -------------------------------------------------
def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    mcfg = cfg.get("model", {})

    model = CrossDomainDegradationTransfer(
        deg_dim=mcfg.get("deg_dim", 256),
        content_dim=mcfg.get("content_dim", 512),
        n_degradation_types=mcfg.get("n_degradation_types", 8),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model


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


def finetune_steps(model, loader, criterion, optimizer, device, steps):
    model.train()
    it = iter(loader)
    for _ in tqdm(range(steps), desc="Few-shot fine-tuning"):
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


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--ft_steps", type=int, default=300)
    ap.add_argument("--ft_lr", type=float, default=5e-5)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------
    # Load source-specific checkpoint
    # -------------------------------------------------
    if "checkpoints" not in cfg or args.source not in cfg["checkpoints"]:
        raise ValueError(f"No checkpoint for source={args.source}")

    ckpt_path = cfg["checkpoints"][args.source]
    image_size=cfg.get("image_size", 64)
    model = load_model_from_ckpt(ckpt_path, device)

    criterion = ICMLLoss(
        recon_weight=cfg["loss"]["recon_weight"],
        kl_weight=cfg["loss"]["kl_weight"],
        disentangle_weight=cfg["loss"]["disentangle_weight"],
        domain_weight=cfg["loss"]["domain_weight"],
    )

    # -------------------------------------------------
    # Data
    # -------------------------------------------------
    _, fewshot_loader, test_loader = create_cross_domain_pairs(
        source_domain=args.source,
        target_domain=args.target,
        data_root=cfg["data_root"],
        n_shots=args.shots,        batch_size=cfg["batch_size"],
        image_size=image_size,
    )

    # -------------------------------------------------
    # Experiment dir
    # -------------------------------------------------
    exp_name = f"fewshot_{args.source}_to_{args.target}_shots{args.shots}"
    exp_dir = Path(cfg["output_dir"]) / f"{exp_name}_{datetime.now():%Y%m%d_%H%M%S}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Zero-shot
    # -------------------------------------------------
    zero = evaluate(model, test_loader, device)
    print(f"[Zero-shot] {args.source} → {args.target}: {zero}")

    # -------------------------------------------------
    # Few-shot adaptation
    # -------------------------------------------------
    if args.shots > 0:
        opt = optim.Adam(model.parameters(), lr=args.ft_lr)
        finetune_steps(model, fewshot_loader, criterion, opt, device, args.ft_steps)

    final = evaluate(model, test_loader, device)
    print(f"[Few-shot={args.shots}] {args.source} → {args.target}: {final}")

    # Save results to CSV
    csv_path = Path(cfg["output_dir"]) / "fewshot_cross_results.csv"
    is_new = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "target", "shots", "psnr", "ssim"]
        )
        if is_new:
            writer.writeheader()
        writer.writerow({
            "source": args.source,
            "target": args.target,
            "shots": args.shots,
            "psnr": final["psnr"],
            "ssim": final["ssim"],
        })

    # Save
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "source": args.source,
            "target": args.target,
            "shots": args.shots,
            "zero": zero,
            "final": final,
        },
        exp_dir / "model_final.pth",
    )

    print(f"Saved results to {exp_dir}")



if __name__ == "__main__":
    main()
