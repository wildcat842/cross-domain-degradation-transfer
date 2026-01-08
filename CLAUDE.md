# CLAUDE.md

Cross-Domain Degradation Transfer Learning - ICML 2026 Submission

## Project Overview

이미지 복원을 위한 Cross-Domain Degradation Transfer Learning 프레임워크.
열화(degradation)와 콘텐츠를 분리(disentangle)하여 도메인 간 전이 학습을 가능하게 함.

### 핵심 Contributions
1. **Degradation-Content Disentanglement**: 열화와 콘텐츠 표현을 분리
2. **Cross-Domain Transfer**: 도메인 불변 열화 표현을 통한 전이 학습

## Tech Stack

- **Framework**: PyTorch >= 2.0.0
- **Language**: Python 3.8+
- **Key Libraries**: torchvision, numpy, pyyaml, tensorboard, scikit-learn, pandas, matplotlib

## Project Structure

```
work/
├── src/
│   ├── models/
│   │   ├── encoders.py      # DegradationEncoder, ContentEncoder
│   │   ├── decoder.py       # CrossDomainDecoder, AdaIN
│   │   └── cddt.py          # 전체 모델 (CrossDomainDegradationTransfer)
│   ├── losses/
│   │   └── icml_loss.py     # ICMLLoss (HSIC, KL, Reconstruction)
│   ├── data/
│   │   ├── datasets.py      # ImageNetC, LDCT, DIBCO, FMD 데이터셋
│   │   └── loader.py        # Multi-domain loader, cross-domain pairs
│   └── utils/
│       ├── metrics.py       # PSNR, SSIM, LPIPS
│       └── visualization.py # t-SNE, training curves, result grids
├── configs/
│   ├── default.yaml              # 기본 설정
│   ├── cross_domain_transfer.yaml # 크로스 도메인 실험
│   └── ablation.yaml             # Ablation study
├── scripts/
│   ├── train.py             # 학습 스크립트 (multi-domain, cross-domain)
│   └── evaluate.py          # 평가 스크립트 (PSNR/SSIM matrix, t-SNE)
├── paper/
│   ├── main.tex             # ICML 2026 논문 템플릿
│   ├── references.bib       # 참고문헌
│   ├── tables/              # 결과 테이블 (LaTeX)
│   └── figures/             # 논문 그림
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   ├── test_models.py       # 모델 유닛 테스트
│   ├── test_losses.py       # 손실 함수 테스트
│   ├── test_data.py         # 데이터 로더 테스트
│   └── test_integration.py  # 통합 테스트
├── notebooks/               # 연구자용 대화형 노트북
│   ├── 01_data_exploration.ipynb    # 데이터 탐색
│   ├── 02_model_visualization.ipynb # 모델 결과 시각화
│   ├── 03_quick_demo.ipynb          # 빠른 데모
│   └── examples/            # 테스트용 이미지
├── experiments/             # 실험 결과 저장
├── requirements.txt
└── pyproject.toml           # pytest 설정
```

## Development Commands

```bash
# 의존성 설치
pip install -r requirements.txt

# Multi-domain 학습 (4개 도메인 동시)
python scripts/train.py --config configs/default.yaml

# Cross-domain transfer 학습 (ImageNet → LDCT)
python scripts/train.py --config configs/default.yaml \
    --source_domain imagenet --target_domain ldct --n_shots 0

# Few-shot adaptation (10-shot)
python scripts/train.py --config configs/default.yaml \
    --source_domain imagenet --target_domain ldct --n_shots 10

# 전체 cross-domain 평가 매트릭스
python scripts/evaluate.py --checkpoint experiments/best.pth \
    --mode all --data_root ./data --output_dir ./results

# t-SNE 시각화 생성
python scripts/evaluate.py --checkpoint experiments/best.pth \
    --mode all --visualize_tsne --output_dir ./results

# 테스트 실행
pytest                           # 전체 테스트
pytest tests/test_models.py      # 모델 테스트만
pytest -v                        # 상세 출력
pytest --cov=src                 # 커버리지 측정
pytest -k "test_forward"         # 특정 테스트만

# Jupyter 노트북 실행
cd notebooks && jupyter notebook
```

## Notebooks (연구자용)

파이프라인 코드를 건드리지 않고 데이터 탐색 및 시각화:

| 노트북 | 설명 |
|--------|------|
| `01_data_exploration.ipynb` | 데이터셋 탐색, 샘플 시각화, 히스토그램 |
| `02_model_visualization.ipynb` | 복원 결과, t-SNE, cross-domain transfer |
| `03_quick_demo.ipynb` | 데이터 없이 합성 이미지로 빠른 테스트 |

## Datasets

| 도메인 | 데이터셋 | 크기 | 다운로드 |
|--------|---------|------|---------|
| Natural | ImageNet-C | 50K | https://github.com/hendrycks/robustness |
| Medical | LDCT-Grand-Challenge | 5K | https://www.aapm.org/grandchallenge/lowdosect/ |
| Document | DIBCO 2019 | 1K | https://vc.ee.duth.gr/dibco/ |
| Microscopy | FMD | 12K | https://github.com/yinhaoz/denoising-fluorescence |

데이터 구조:
```
data/
├── imagenet-c/          # corruption/severity/class/image.jpg
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

## Experiment Matrix

| Source → Target | Zero-shot | 10-shot | 100-shot |
|-----------------|-----------|---------|----------|
| ImageNet → LDCT | ✓ | ✓ | ✓ |
| ImageNet → DIBCO | ✓ | ✓ | ✓ |
| LDCT → Microscopy | ✓ | ✓ | ✓ |
| DIBCO → Cryo-EM | ✓ | ✓ | ✓ |

## Code Style and Conventions

- 한국어 주석 허용 (논문 준비용)
- VAE 기반 열화 표현: `z_d` (continuous + discrete type)
- 콘텐츠 표현: `z_c` (도메인 특화)
- 모델 출력은 dictionary 형태로 반환
- 이미지 범위: [-1, 1] (학습), [0, 1] (평가/시각화)

## Important Notes

### 이론적 가정
- 열화 패턴은 저차원 매니폴드 Z_d에 존재
- 콘텐츠와 열화는 독립: I(z_c; z_d) ≈ 0 (HSIC로 측정)
- 열화 표현은 도메인 불변 (domain adversarial training)

### 손실 함수 구성
1. **Reconstruction Loss**: L1 loss (`recon_weight=1.0`)
2. **KL Divergence**: VAE regularization (`kl_weight=0.01`)
3. **Disentanglement Loss**: HSIC 기반 (`disentangle_weight=0.1`)
4. **Domain Adversarial Loss**: 도메인 불변성 (`domain_weight=0.1`)

### Baseline 비교 대상
- **Supervised**: NAFNet, Restormer, DnCNN
- **Unsupervised Transfer**: CycleGAN, UNIT

### 논문 작성
- LaTeX 템플릿: `paper/main.tex` (ICML 2026 형식)
- 결과 테이블: `paper/tables/`
- 예상 Figure: Framework overview, t-SNE of z_d, Qualitative comparisons
