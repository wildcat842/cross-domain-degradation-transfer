import re
from pathlib import Path
import pandas as pd
import os
import sys

# 스크립트 디렉토리 계산
script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, str(Path(script_dir).parent.parent))

fewshot = pd.read_csv(os.path.join(script_dir, "../../results/fewshot/fewshot_cross_results.csv"))
zeroshot = pd.read_csv(os.path.join(script_dir,"../../results/imagenet_corruption_eval/imagenet_cross_corruption_results.csv"))

ROOT = Path(os.path.join(script_dir, "../../paper/tables"))

# 폴더명 패턴: tb_figs_cddt_noiseonly, tb_figs_cddt_bluronly, tb_figs_cddt_weatheronly
EXPS = {
    "noise": ROOT / "tb_figs_cddt_noiseonly",
    "blur": ROOT / "tb_figs_cddt_bluronly",
    "weather": ROOT / "tb_figs_cddt_weatheronly",
}

def read_tb_summary_csv(folder: Path) -> pd.DataFrame:
    """
    각 폴더 안의 'tag,last_step,last_value,best_min_value,best_step,best_max_value' 형식 CSV를 읽어서
    하나의 long-form 데이터프레임으로 반환.
    """
    # 폴더 내 csv가 1개일 수도, 여러 개일 수도 있어서 전부 읽고 concat
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in: {folder}")

    dfs = []
    for fp in csvs:
        df = pd.read_csv(fp)
        df["file"] = fp.name
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    # 기본 컬럼 정리
    expected = {"tag","last_step","last_value","best_min_value","best_step","best_max_value"}
    missing = expected - set(out.columns)
    if missing:
        raise ValueError(f"{folder} CSV missing columns: {missing}")
    return out


def build_wide_table(long_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    tag를 row로 두고, last/best를 column으로 두는 wide 형태로 변환.
    모델명 컬럼을 붙여서 나중에 merge 가능하게 함.
    """
    # 어떤 실험은 best_min_value가 있고 어떤 실험은 best_max_value가 있고… 해서 둘 다 보존
    wide = long_df.pivot_table(
        index="tag",
        values=["last_step","last_value","best_min_value","best_step","best_max_value"],
        aggfunc="last",
    ).reset_index()

    # 멀티컬럼 평탄화
    wide.columns = ["tag"] + [f"{model_name}__{c}" for c in wide.columns[1:]]
    return wide


# 1) 읽기
long_all = []
for model, folder in EXPS.items():
    df = read_tb_summary_csv(folder)
    df["model"] = model
    long_all.append(df)
long_all = pd.concat(long_all, ignore_index=True)

# 2) wide로 만들고 merge (tag 기준 outer join)
wide_tables = []
for model in EXPS.keys():
    wide_tables.append(build_wide_table(long_all[long_all["model"] == model], model))

merged = wide_tables[0]
for wt in wide_tables[1:]:
    merged = merged.merge(wt, on="tag", how="outer")

# 3) 분석에 유용한 태그만 골라서 별도 뷰 만들기
KEY_TAGS = [
    "train/loss",
    "train/psnr",
    "val/avg_psnr",
    "val/imagenet-noise_psnr",
    "val/imagenet-noise_ssim",
    "val/imagenet-blur_psnr",
    "val/imagenet-blur_ssim",
    "val/imagenet-weather_psnr",
    "val/imagenet-weather_ssim",
]
focus = merged[merged["tag"].isin(KEY_TAGS)].copy()

# 저장
out_dir = ROOT / "merged_reports"
out_dir.mkdir(parents=True, exist_ok=True)
merged.to_csv(out_dir / "tb_summary_merged_all_tags.csv", index=False)
focus.to_csv(out_dir / "tb_summary_merged_key_tags.csv", index=False)

print("Saved:")
print(" -", out_dir / "tb_summary_merged_all_tags.csv")
print(" -", out_dir / "tb_summary_merged_key_tags.csv")

# 4) 콘솔에서 바로 비교가 보이도록 'same-domain 성능표' 형태로 재구성
def pick(df, tag, col):
    row = df[df["tag"] == tag]
    return None if row.empty else row.iloc[0][col]

same_rows = []
for model in EXPS.keys():
    # 각 모델에 대해 "자기 도메인"의 val 성능을 뽑고 싶음
    # 예: noise-only 모델이면 val/imagenet-noise_psnr/ssim이 핵심
    t_psnr = f"val/imagenet-{model}_psnr"
    t_ssim = f"val/imagenet-{model}_ssim"

    row = {
        "model": model,
        "val_same_psnr_last": pick(focus, t_psnr, f"{model}__last_value"),
        "val_same_psnr_best": pick(focus, t_psnr, f"{model}__best_max_value"),
        "val_same_psnr_best_step": pick(focus, t_psnr, f"{model}__best_step"),
        "val_same_ssim_last": pick(focus, t_ssim, f"{model}__last_value"),
        "val_same_ssim_best": pick(focus, t_ssim, f"{model}__best_max_value"),
        "val_same_ssim_best_step": pick(focus, t_ssim, f"{model}__best_step"),
        "val_avg_psnr_last": pick(focus, "val/avg_psnr", f"{model}__last_value"),
        "val_avg_psnr_best": pick(focus, "val/avg_psnr", f"{model}__best_max_value"),
        "train_psnr_last": pick(focus, "train/psnr", f"{model}__last_value"),
        "train_loss_last": pick(focus, "train/loss", f"{model}__last_value"),
    }
    same_rows.append(row)

same_table = pd.DataFrame(same_rows)
same_table.to_csv(out_dir / "tb_same_domain_comparison.csv", index=False)
print(" -", out_dir / "tb_same_domain_comparison.csv")
print("\nSame-domain comparison (preview):")
print(same_table.to_string(index=False))


import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT = Path(os.path.join(script_dir, "../../paper/tables")) / "merged_reports"

df = pd.read_csv(ROOT / "tb_same_domain_comparison.csv")

# PSNR
plt.figure()
plt.bar(df["model"], df["val_same_psnr_best"])
plt.title("Best Same-domain PSNR (per corruption-only model)")
plt.ylabel("PSNR")
plt.tight_layout()
plt.savefig(ROOT / "same_domain_psnr_best_bar.png", dpi=200)
plt.close()

# SSIM
plt.figure()
plt.bar(df["model"], df["val_same_ssim_best"])
plt.title("Best Same-domain SSIM (per corruption-only model)")
plt.ylabel("SSIM")
plt.tight_layout()
plt.savefig(ROOT / "same_domain_ssim_best_bar.png", dpi=200)
plt.close()

print("Saved plots to:", ROOT)
