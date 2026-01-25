import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 스크립트 디렉토리 계산
script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, str(Path(script_dir).parent.parent))

fewshot = pd.read_csv(os.path.join(script_dir, "../../results/fewshot/fewshot_cross_results.csv"))
zeroshot = pd.read_csv(os.path.join(script_dir,"../../results/imagenet_corruption_eval/imagenet_cross_corruption_results.csv"))

merged = fewshot.merge(
    zeroshot,
    on=["source", "target"],
    suffixes=("_few", "_zero")
)

merged["delta_psnr"] = merged["psnr_few"] - merged["psnr_zero"]

outdir = Path(os.path.join(script_dir, "../../paper/figures"))
outdir.mkdir(exist_ok=True)

for shots in [10, 50, 100]:
    df = fewshot[fewshot["shots"] == shots]
    pivot = df.pivot(index="source", columns="target", values="psnr")

    plt.figure(figsize=(6,5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        square=True,
        cbar=True,
    )
    plt.title(f"Few-shot Transfer PSNR (shots={shots})")
    plt.tight_layout()
    plt.savefig(outdir / f"fewshot_psnr_{shots}.png", dpi=200)
    plt.close()

plt.figure(figsize=(8,4))
sns.barplot(
    data=merged,
    x="shots",
    y="delta_psnr",
    hue="source",
)
plt.axhline(0, color="gray", linestyle="--")
plt.ylabel("ΔPSNR (Few-shot − Zero-shot)")
plt.title("Few-shot Adaptation Gain across Corruptions")
plt.tight_layout()
plt.savefig(outdir / "delta_psnr_bar.png", dpi=200)
plt.close()