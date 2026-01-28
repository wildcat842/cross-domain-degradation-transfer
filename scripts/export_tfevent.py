import os
import pandas as pd
import matplotlib.pyplot as plt

# 스크립트 디렉토리 계산
script_dir = os.path.dirname(os.path.abspath(__file__))

# 1) (필요시) 설치
# pip install tensorboard

from tensorboard.backend.event_processing import event_accumulator

################################# 체크포인트가 있는 파일 경로로 바꾸세요 #################################
#EVENT_PATH = os.path.join(script_dir, "../experiments/cddt_weatheronly_imagenetW/cddt_weatheronly_imagenetW_20260125_192212/logs/events.out.tfevents.1769336538.server179-96.2018133.0")
EVENT_PATH = os.path.join(script_dir, "../experiments/dncnn_weatheronly_imagenet/dncnn_weatheronly_imagenet_20260127_183037/logs/events.out.tfevents.1769506243.server179-96.3282682.0")


####################### 폴더 명을 바꾸세요 #########
out_dir = os.path.join(script_dir, "../paper/tables/tb_figs_dcnn_weatheronly")


# 2) 이벤트 로드
ea = event_accumulator.EventAccumulator(
    EVENT_PATH,
    size_guidance={
        event_accumulator.SCALARS: 0,
        event_accumulator.IMAGES: 0,
        event_accumulator.HISTOGRAMS: 0,
        event_accumulator.TENSORS: 0,
        event_accumulator.AUDIO: 0,
    },
)
ea.Reload()

tags = ea.Tags()
print("== Tags ==")
print(tags)

# 3) Scalar를 tidy 포맷으로 덤프: [tag, step, wall_time, value]
rows = []
for tag in tags.get("scalars", []):
    for ev in ea.Scalars(tag):
        rows.append({"tag": tag, "step": ev.step, "wall_time": ev.wall_time, "value": ev.value})

df = pd.DataFrame(rows).sort_values(["tag", "step"]).reset_index(drop=True)


os.makedirs(out_dir, exist_ok=True)

# 4) CSV 저장 (tidy / wide) 

df.to_csv(os.path.join(out_dir, "tb_scalars_tidy.csv"), index=False)

wide = df.pivot_table(index="step", columns="tag", values="value").reset_index().sort_values("step")
wide.to_csv(os.path.join(out_dir, "tb_scalars_wide.csv"), index=False)

# 5) Figure 저장 (tag별 1개씩)

for tag in tags.get("scalars", []):
    sub = df[df["tag"] == tag]
    plt.figure()
    plt.plot(sub["step"], sub["value"])
    plt.xlabel("step")
    plt.ylabel(tag)
    plt.title(tag)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, tag.replace("/", "_") + ".png"), dpi=200)
    plt.close()

# 6) (옵션) 전체 스칼라를 한 장에 “정규화”해서 비교용 Figure 생성
import numpy as np

plt.figure()
for tag in tags.get("scalars", []):
    sub = df[df["tag"] == tag].sort_values("step")
    x = sub["step"].to_numpy()
    y = sub["value"].to_numpy()
    if np.nanmax(y) != np.nanmin(y):
        y = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y))
    else:
        y = y * 0
    plt.plot(x, y, label=tag)

plt.xlabel("step")
plt.ylabel("normalized value (per-tag)")
plt.title("All Scalars (normalized)")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "all_scalars_normalized.png"), dpi=200)
plt.close()

# 7) 논문용 요약 테이블(마지막 값 / 최고(또는 최저) 값 / 해당 step)
summary_rows = []
for tag in tags.get("scalars", []):
    sub = df[df["tag"] == tag].sort_values("step")
    last = sub.iloc[-1]

    # loss면 낮을수록 좋다고 가정, 나머지는 높을수록 좋다고 가정(원하면 규칙 바꾸세요)
    if "loss" in tag.lower():
        best = sub.loc[sub["value"].idxmin()]
        best_key = "best_min_value"
    else:
        best = sub.loc[sub["value"].idxmax()]
        best_key = "best_max_value"

    summary_rows.append({
        "tag": tag,
        "last_step": int(last["step"]),
        "last_value": float(last["value"]),
        best_key: float(best["value"]),
        "best_step": int(best["step"]),
    })

summary = pd.DataFrame(summary_rows).sort_values("tag").reset_index(drop=True)
summary.to_csv(os.path.join(out_dir, "tb_scalars_summary.csv"), index=False)

print("\nSaved:")
print(f"- tb_scalars_tidy.csv in {out_dir}")
print(f"- tb_scalars_wide.csv in {out_dir}")
print(f"- tb_scalars_summary.csv in {out_dir}")
print(f"- Figures in {out_dir}/")
