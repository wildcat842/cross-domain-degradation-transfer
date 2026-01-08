# Cross-Domain Degradation Transfer Learning

Cross-Domain Degradation Transfer Learning for Universal Image Restoration - ICML 2026 Submission

## Overview

이미지 복원을 위한 Cross-Domain Degradation Transfer Learning 프레임워크입니다. 열화(degradation)와 콘텐츠를 분리(disentangle)하여 도메인 간 전이 학습을 가능하게 합니다.

### Key Features

- **Degradation-Content Disentanglement**: VAE 기반 열화 표현 학습
- **Cross-Domain Transfer**: 도메인 불변 열화 표현을 통한 zero-shot/few-shot 전이
- **Multi-Domain Support**: ImageNet-C, LDCT, DIBCO, FMD 데이터셋 지원

## Quick Start

### 1. 저장소 클론

```bash
git clone https://github.com/wildcat842/cross-domain-degradation-transfer.git
cd cross-domain-degradation-transfer
```

### 2. 가상환경 설정

#### Windows
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate
```

#### macOS / Linux
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/bin/activate
```

#### Conda 사용 시
```bash
# Conda 환경 생성 (Python 3.10 ~ 3.12 지원)
conda create -n cddt python=3.12
conda activate cddt
```

### 3. 의존성 설치

```bash
# 기본 설치
pip install -r requirements.txt

# PyTorch (CUDA 지원 시)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# PyTorch (CPU only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. VS Code 설정 (선택)

IntelliSense 및 자동 venv 활성화를 위해 `.vscode/settings.json` 생성:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "jupyter.defaultKernel": "Python (venv)"
}
```

> Windows: `venv/bin/python` → `venv/Scripts/python.exe`

### 5. 설치 확인

```bash
# 테스트 실행으로 설치 확인
pytest tests/test_data.py -v
```

## Git 사용 가이드

### 현재 상태 확인 (먼저 실행 권장)

```bash
git status          # 수정된 파일 확인
git branch          # 현재 브랜치 확인
```

### 원격 저장소에서 최신 코드 받기

#### 1. 로컬에 수정한 게 없을 때 (가장 간단)
```bash
git pull
```

#### 2. 로컬에 수정한 게 있고, 유지하고 싶을 때 (권장)
```bash
git pull --rebase
```

충돌 발생 시:
```bash
git status              # 충돌 파일 확인
# 충돌 파일 수정 후
git add <파일>
git rebase --continue
```

#### 3. 로컬 변경사항을 잠시 치워두고 업데이트
```bash
git stash               # 변경사항 임시 저장
git pull
git stash pop           # 변경사항 복원
```

#### 4. 다운받기만 하고 적용은 나중에
```bash
git fetch               # 코드 안 바뀜, 변경사항만 확인 가능
git diff HEAD origin/main   # 변경 내용 확인
git merge origin/main   # 적용
```

#### 5. 로컬 변경 전부 버리고 최신으로 맞추기 (⚠️ 주의: 되돌릴 수 없음)
```bash
git fetch origin
git reset --hard origin/master
```

### 추천 워크플로우

```bash
git status
git pull --rebase
```

문제 생기면:
```bash
git rebase --abort      # rebase 취소
```

## Project Structure

```
cross-domain-degradation-transfer/
│
├── src/                          # 소스 코드
│   ├── models/                   # 모델 정의
│   │   ├── encoders.py          # DegradationEncoder, ContentEncoder
│   │   ├── decoder.py           # CrossDomainDecoder, AdaIN
│   │   └── cddt.py              # 전체 모델 (CrossDomainDegradationTransfer)
│   │
│   ├── losses/                   # 손실 함수
│   │   └── icml_loss.py         # ICMLLoss (HSIC, KL, Reconstruction)
│   │
│   ├── data/                     # 데이터 로딩
│   │   ├── datasets.py          # 도메인별 데이터셋 클래스
│   │   └── loader.py            # Multi-domain loader
│   │
│   └── utils/                    # 유틸리티
│       ├── metrics.py           # PSNR, SSIM, LPIPS
│       └── visualization.py     # t-SNE, 시각화 도구
│
├── configs/                      # 설정 파일
│   ├── default.yaml             # 기본 설정
│   ├── cross_domain_transfer.yaml
│   └── ablation.yaml
│
├── scripts/                      # 실행 스크립트
│   ├── train.py                 # 학습
│   └── evaluate.py              # 평가
│
├── tests/                        # 테스트
│   ├── conftest.py              # Pytest fixtures
│   ├── test_models.py           # 모델 테스트
│   ├── test_losses.py           # 손실 함수 테스트
│   ├── test_data.py             # 데이터 로더 테스트
│   └── test_integration.py      # 통합 테스트
│
├── notebooks/                    # 연구자용 노트북
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_visualization.ipynb
│   ├── 03_quick_demo.ipynb
│   └── examples/                # 테스트 이미지
│
├── paper/                        # 논문 관련
│   ├── main.tex                 # ICML 2026 템플릿
│   ├── references.bib
│   └── tables/
│
├── experiments/                  # 실험 결과 (git ignored)
├── data/                         # 데이터셋 (git ignored)
│
├── requirements.txt
├── pyproject.toml               # pytest 설정
├── CLAUDE.md                    # Claude Code 가이드
└── README.md
```

## Usage

### Training

```bash
# Multi-domain 학습 (4개 도메인 동시)
python scripts/train.py --config configs/default.yaml

# Cross-domain transfer (ImageNet → LDCT, zero-shot)
python scripts/train.py --config configs/default.yaml \
    --source_domain imagenet --target_domain ldct --n_shots 0

# Few-shot adaptation (10-shot)
python scripts/train.py --config configs/default.yaml \
    --source_domain imagenet --target_domain ldct --n_shots 10
```

### Evaluation

```bash
# 전체 cross-domain 평가 매트릭스
python scripts/evaluate.py --checkpoint experiments/best.pth \
    --mode all --data_root ./data --output_dir ./results

# 특정 도메인 쌍 평가
python scripts/evaluate.py --checkpoint experiments/best.pth \
    --mode cross_domain --source_domain imagenet --target_domain ldct

# t-SNE 시각화 생성
python scripts/evaluate.py --checkpoint experiments/best.pth \
    --mode all --visualize_tsne --output_dir ./results
```

### Testing

```bash
# 전체 테스트
pytest

# 상세 출력
pytest -v

# 특정 테스트 파일
pytest tests/test_models.py

# 특정 테스트 클래스
pytest tests/test_models.py::TestDegradationEncoder

# 특정 테스트 함수
pytest -k "test_output_shape"

# 커버리지 측정
pytest --cov=src --cov-report=html
```

### Notebooks (Interactive Exploration)

파이프라인 코드를 건드리지 않고 데이터 탐색 및 시각화:

| 노트북 | 설명 | 데이터 필요 |
|--------|------|-------------|
| `01_data_exploration.ipynb` | 데이터셋 탐색, 샘플 시각화, 히스토그램 | O |
| `02_model_visualization.ipynb` | 복원 결과, t-SNE, cross-domain transfer | O |
| `03_quick_demo.ipynb` | 합성 이미지로 빠른 모델 테스트 | X |

#### Jupyter 설정 (VS Code)

```bash
# 1. venv에 Jupyter 설치
pip install ipykernel jupyter
python -m ipykernel install --user --name venv --display-name "Python (venv)"

# 2. VS Code에서 커널 선택
#    - .ipynb 파일 열기 → 우측 상단 "Select Kernel" → venv 선택

# 3. 확인 (노트북 셀에서)
import sys
print(sys.executable)  # venv 경로가 출력되면 정상
```

> 커널이 안 보이면: venv 활성화 후 VS Code 재시작 (`source venv/bin/activate && code .`)

## Datasets

| Domain | Dataset | Size | Download |
|--------|---------|------|----------|
| Natural | ImageNet-C | 50K | [Link](https://github.com/hendrycks/robustness) |
| Medical | LDCT-Grand-Challenge | 5K | [Link](https://www.aapm.org/grandchallenge/lowdosect/) |
| Document | DIBCO 2019 | 20 | [Official](https://vc.ee.duth.gr/dibco2019/) / [GitHub](https://github.com/tanmayGIT/DIBCO_2019_All) |
| Microscopy | FMD | 12K | [Link](https://github.com/yinhaoz/denoising-fluorescence) |

### Data Directory Structure

```
data/
├── imagenet-c/
│   └── {corruption}/{severity}/{class}/{image}.jpg
├── ldct/
│   ├── train/
│   │   ├── low_dose/
│   │   └── full_dose/
│   └── test/
├── dibco/
│   └── 2019/
│       ├── imgs/
│       └── gt/
└── fmd/
    └── Confocal_BPAE/
        ├── noisy/
        └── gt/
```

## Model Architecture

```
┌─────────────┐     ┌─────────────────┐
│  Degraded   │────▶│ DegradationEnc  │────▶ z_d (256-dim, domain-invariant)
│   Image     │     │     (VAE)       │
│             │     └─────────────────┘
│             │
│             │     ┌─────────────────┐     ┌─────────────┐
│             │────▶│  ContentEncoder │────▶│   Decoder   │────▶ Restored
└─────────────┘     └─────────────────┘     │   (AdaIN)   │      Image
                           z_c              └─────────────┘
                    (512-dim spatial)            ▲
                                                 │ -z_d (remove degradation)
                                                 │
```

## Loss Functions

| Loss | Weight | Description |
|------|--------|-------------|
| Reconstruction | 1.0 | L1 loss between restored and clean |
| KL Divergence | 0.01 | VAE regularization for z_d |
| Disentanglement | 0.1 | HSIC-based independence (z_c ⊥ z_d) |
| Domain Adversarial | 0.1 | Domain-invariant z_d |

## Requirements

- Python 3.10 - 3.12 (3.12 권장)
- PyTorch >= 2.4.0
- CUDA >= 11.8 (optional, for GPU)

## Notes

### Image Range Convention
- Training: `[-1, 1]` (normalized with mean=0.5, std=0.5)
- Evaluation/Visualization: `[0, 1]`

### GPU Memory
- Default batch size 16 requires ~8GB VRAM
- Reduce batch size for smaller GPUs:
  ```bash
  python scripts/train.py --config configs/default.yaml  # Edit batch_size in yaml
  ```

### Reproducibility
- Set random seed in config for reproducible results
- Results may vary slightly between CPU and GPU

## Citation

```bibtex
@inproceedings{cddt2026,
  title={Cross-Domain Degradation Transfer for Universal Image Restoration},
  author={Anonymous},
  booktitle={ICML},
  year={2026}
}
```

## License

MIT License

## Acknowledgments

- [NAFNet](https://github.com/megvii-research/NAFNet)
- [Restormer](https://github.com/swz30/Restormer)
- [ImageNet-C](https://github.com/hendrycks/robustness)
