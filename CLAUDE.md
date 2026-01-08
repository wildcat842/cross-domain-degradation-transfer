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
- **Key Libraries**: torchvision, numpy, pyyaml, tensorboard

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
│   ├── data/                # 데이터 로딩 유틸리티
│   └── utils/               # 헬퍼 함수
├── configs/
│   └── default.yaml         # 기본 설정
├── scripts/
│   ├── train.py             # 학습 스크립트
│   └── evaluate.py          # 평가 스크립트
├── experiments/             # 실험 결과 저장
└── requirements.txt
```

## Development Commands

```bash
# 의존성 설치
pip install -r requirements.txt

# 학습 실행
python scripts/train.py --config configs/default.yaml

# 평가 실행
python scripts/evaluate.py --checkpoint <path> --data_dir <path>
```

## Code Style and Conventions

- 한국어 주석 허용 (논문 준비용)
- VAE 기반 열화 표현: `z_d` (continuous + discrete type)
- 콘텐츠 표현: `z_c` (도메인 특화)
- 모델 출력은 dictionary 형태로 반환

## Important Notes

### 이론적 가정
- 열화 패턴은 저차원 매니폴드 Z_d에 존재
- 콘텐츠와 열화는 독립: I(z_c; z_d) ≈ 0 (HSIC로 측정)
- 열화 표현은 도메인 불변 (domain adversarial training)

### 손실 함수 구성
1. **Reconstruction Loss**: L1 loss
2. **KL Divergence**: VAE regularization
3. **Disentanglement Loss**: HSIC 기반
4. **Domain Adversarial Loss**: 도메인 불변성
