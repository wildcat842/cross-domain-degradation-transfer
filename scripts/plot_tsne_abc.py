import re
from pathlib import Path
import torch
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_multi_domain_loader, create_cross_domain_pairs
from src.models import CrossDomainDegradationTransfer
from src.utils.visualization import plot_tsne_degradation


# -----------------------------
# Config 
# -----------------------------
DATA_ROOT = "./data"
OUT_DIR = Path("./paper/figures/tsne")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_DOMAINS = ["imagenet-noise", "imagenet-blur", "imagenet-weather"]

IMAGE_SIZE = 64        # TinyImageNet 기반이면 64 권장
BATCH_SIZE = 16
NUM_WORKERS = 4
MAX_SAMPLES = 600      # 도메인당 점 개수


# -----------------------------
# Helpers
# -----------------------------
def load_ckpt_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    mcfg = cfg.get("model", {})

    model = CrossDomainDegradationTransfer(
        deg_dim=mcfg.get("deg_dim", 256),
        content_dim=mcfg.get("content_dim", 512),
        n_degradation_types=mcfg.get("n_degradation_types", 8),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def load_final_model_from_fewshot_pth(model_pth: str, base_ckpt: str, device: torch.device):
    """
    few-shot 결과 폴더의 model_final.pth에는 config가 없을 수 있으므로,
    base checkpoint의 config로 모델을 만들고 state_dict만 덮어쓴다.
    """
    base = torch.load(base_ckpt, map_location=device)
    cfg = base.get("config", {})
    mcfg = cfg.get("model", {})

    model = CrossDomainDegradationTransfer(
        deg_dim=mcfg.get("deg_dim", 256),
        content_dim=mcfg.get("content_dim", 512),
        n_degradation_types=mcfg.get("n_degradation_types", 8),
    ).to(device)

    w = torch.load(model_pth, map_location=device, weights_only=False)
    model.load_state_dict(w["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_zd(model, loader, device, max_samples=600):
    zs, ts = [], []
    n = 0
    model.eval()

    for batch in loader:
        degraded = batch["degraded"].to(device)
        out = model(degraded)

        # z_d
        if "z_d" not in out:
            raise KeyError("Model output does not contain 'z_d'. Please ensure forward returns it.")
        z = out["z_d"].detach().cpu()
        zs.append(z)

        # deg_type (optional)
        if "deg_type" in out:
            dt = out["deg_type"]
            if dt.dim() == 2:
                dt = dt.argmax(dim=1)
            ts.append(dt.detach().cpu())
        else:
            ts.append(torch.full((z.size(0),), -1))

        n += z.size(0)
        if n >= max_samples:
            break

    z_all = torch.cat(zs, dim=0)[:max_samples]
    t_all = torch.cat(ts, dim=0)[:max_samples]
    return z_all, t_all


# -----------------------------
# Figure A: multi-domain model, domains=noise/blur/weather
# -----------------------------
def make_figure_A(multidomain_ckpt: str, device):
    model = load_ckpt_model(multidomain_ckpt, device)

    loaders = create_multi_domain_loader(
        domains=IMAGENET_DOMAINS,
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
        split="test",
    )

    z_d_dict, deg_dict = {}, {}
    for d in IMAGENET_DOMAINS:
        z, t = collect_zd(model, loaders[d], device, MAX_SAMPLES)
        z_d_dict[d] = z
        deg_dict[d] = t

    plot_tsne_degradation(
        z_d_dict=z_d_dict,
        deg_types=deg_dict,
        save_path=str(OUT_DIR / "FigureA_multidomain_domains_tsne.png"),
        perplexity=30,
    )
    print("Saved Figure A:", OUT_DIR / "FigureA_multidomain_domains_tsne.png")


# -----------------------------
# Figure B: zero-shot vs few-shot latent shift on SAME target
# (shots=10/50/100 각각 저장)
# -----------------------------
def make_figure_B(
    source_domain: str,
    target_domain: str,
    source_ckpt: str,
    fewshot_results_root: str,
    device,
):
    # 1) target test loader 고정
    _, _, test_loader = create_cross_domain_pairs(
        source_domain=source_domain,
        target_domain=target_domain,
        data_root=DATA_ROOT,
        n_shots=0,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
    )

    # 2) zero-shot model
    zero_model = load_ckpt_model(source_ckpt, device)
    z0, t0 = collect_zd(zero_model, test_loader, device, MAX_SAMPLES)

    # 3) few-shot models (model_final.pth가 있는 것만)
    root = Path(fewshot_results_root)
    pattern = re.compile(rf"fewshot_{re.escape(source_domain)}_to_{re.escape(target_domain)}_shots(\d+)_\d{{8}}_\d{{6}}")

    # shots별로 최신 폴더(시간 가장 큰 것) 하나 선택
    shot_to_dir = {}
    for p in root.glob(f"fewshot_{source_domain}_to_{target_domain}_shots*_*"):
        m = pattern.match(p.name)
        if not m:
            continue
        shots = int(m.group(1))
        if shots not in [10, 50, 100]:
            continue
        if shots not in shot_to_dir or p.name > shot_to_dir[shots].name:
            shot_to_dir[shots] = p

    for shots in [10, 50, 100]:
        if shots not in shot_to_dir:
            print(f"[Figure B] shots={shots} 폴더를 못 찾았어. (또는 패턴이 다름)")
            continue

        p = shot_to_dir[shots]
        model_pth = p / "model_final.pth"
        if not model_pth.exists():
            print(f"[Figure B] {model_pth} 없음 → finetune_fewshot.py에서 model_final.pth 저장이 필요")
            continue

        few_model = load_final_model_from_fewshot_pth(str(model_pth), source_ckpt, device)
        z1, t1 = collect_zd(few_model, test_loader, device, MAX_SAMPLES)

        plot_tsne_degradation(
            z_d_dict={"zero-shot": z0, f"few-shot({shots})": z1},
            deg_types={"zero-shot": t0, f"few-shot({shots})": t1},
            save_path=str(OUT_DIR / f"FigureB_zero_vs_few_{source_domain}_to_{target_domain}_shots{shots}.png"),
            perplexity=30,
        )
        print("Saved Figure B:", OUT_DIR / f"FigureB_zero_vs_few_{source_domain}_to_{target_domain}_shots{shots}.png")


# -----------------------------
# Figure C: same target, compare source-only models (bias effect)
# 예) target=imagenet-blur 고정, source-only ckpt 3개 비교
# -----------------------------
def make_figure_C(
    target_domain: str,
    source_ckpts: dict,
    device,
):
    # target test loader 고정 (source는 아무거나 넣어도 target loader만 쓰면 됨)
    _, _, test_loader = create_cross_domain_pairs(
        source_domain="imagenet-noise",
        target_domain=target_domain,
        data_root=DATA_ROOT,
        n_shots=0,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
    )

    z_d_dict, deg_dict = {}, {}
    for src_name, ckpt in source_ckpts.items():
        model = load_ckpt_model(ckpt, device)
        z, t = collect_zd(model, test_loader, device, MAX_SAMPLES)
        z_d_dict[f"src={src_name}"] = z
        deg_dict[f"src={src_name}"] = t

    plot_tsne_degradation(
        z_d_dict=z_d_dict,
        deg_types=deg_dict,
        save_path=str(OUT_DIR / f"FigureC_source_bias_on_{target_domain}.png"),
        perplexity=30,
    )
    print("Saved Figure C:", OUT_DIR / f"FigureC_source_bias_on_{target_domain}.png")


# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open("configs/plot_tsne_abc.yaml", "r") as f:
        config = yaml.safe_load(f)

    ckpts = config["ckpts"]
    CKPT_MULTIDOMAIN = ckpts["multidomain"]
    CKPT_NOISE_ONLY = ckpts["noise_only"]
    CKPT_BLUR_ONLY = ckpts["blur_only"]
    CKPT_WEATHER_ONLY = ckpts["weather_only"]
    FEWSHOT_ROOT = config["fewshot_root"]

    # Figure A
    make_figure_A(CKPT_MULTIDOMAIN, device)

    # Figure B (대표 pair 하나 골라서 그리는 걸 추천)
    # 예: blur -> noise (너가 많이 봤던 케이스)
    make_figure_B(
        source_domain="imagenet-blur",
        target_domain="imagenet-noise",
        source_ckpt=CKPT_BLUR_ONLY,
        fewshot_results_root=FEWSHOT_ROOT,
        device=device,
    )

    # Figure C (같은 target에서 source-only 모델 3개 비교)
    make_figure_C(
        target_domain="imagenet-blur",
        source_ckpts={
            "noise-only": CKPT_NOISE_ONLY,
            "blur-only": CKPT_BLUR_ONLY,
            "weather-only": CKPT_WEATHER_ONLY,
        },
        device=device,
    )


if __name__ == "__main__":
    main()
