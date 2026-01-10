# Data Directory

데이터셋을 아래 구조에 맞춰 저장하세요.

## 현재 데이터 상태

| 데이터셋 | 이미지 수 | 설명 |
|----------|-----------|------|
| **ImageNet-C** | 110,000 | 11 corruptions x 5 severities x 200 classes (64x64) |
| **LDCT** | 3,584 paired | full_dose + synthetic low_dose (362x362) |
| **DIBCO** | 10 | 문서 이미지 + GT binarization |
| **FMD** | 3,000 | 현미경 이미지 (50 noisy → 1 clean) |

## 폴더 구조

```
data/
├── imagenet-c/              # Natural images (corruption)
│   └── Tiny-ImageNet-C/
│       └── {corruption}/{severity}/{class}/{image}.JPEG
│       # corruption: brightness, contrast, defocus_blur, elastic_transform,
│       #             frost, glass_blur, impulse_noise, jpeg_compression,
│       #             motion_blur, pixelate, shot_noise (11종)
│       # severity: 1-5
│
├── ldct/                    # Low-Dose CT (Medical)
│   ├── train/
│   │   ├── low_dose/        # 열화 이미지
│   │   └── full_dose/       # Clean 이미지
│   └── test/
│       ├── low_dose/        # 3,584 synthetic noisy images
│       └── full_dose/       # 3,584 ground truth images (from HDF5)
│
├── dibco/                   # Document Binarization
│   └── 2019/
│       ├── imgs/            # 10 열화 문서 이미지
│       └── gt/              # 10 Ground truth (binarized)
│
└── fmd/                     # Fluorescence Microscopy
    ├── noisy/               # 3,000 images ({type}_{capture}_{file}.png)
    └── clean/               # 60 images ({type}_{capture}.png)
```

## 다운로드 링크

| 도메인 | 데이터셋 | 크기 | 링크 |
|--------|---------|------|------|
| Natural | Tiny-ImageNet-C | 110K | [GitHub](https://github.com/hendrycks/robustness) |
| Medical | LDCT | 3.5K | [TCIA](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/) |
| Document | DIBCO 2019 | 10 | [Official](https://vc.ee.duth.gr/dibco2019/) / [GitHub](https://github.com/tanmayGIT/DIBCO_2019_All) |
| Microscopy | FMD | 3K | [GitHub](https://github.com/yinhaoz/denoising-fluorescence) |

## 데이터 특징

### ImageNet-C (Tiny-ImageNet-C)
- **Clean 이미지 없음**: Self-supervised 또는 다른 corruption을 target으로 사용
- **11가지 corruption**: noise(2), blur(3), weather(2), digital(4)
- **5단계 severity**: 1(약함) ~ 5(강함)

### LDCT
- **Synthetic low_dose**: ground_truth에 Gaussian noise (σ=25) 추가
- **원본 observation 데이터**: Zenodo에서 더 이상 직접 다운로드 불가

### FMD
- **50:1 비율**: 50개 noisy 이미지가 1개 clean 이미지에 매핑
- **3가지 현미경 타입**: Confocal_BPAE (B/G/R)

## 확인

노트북에서 데이터 존재 여부 확인:

```python
from pathlib import Path

DATA_ROOT = Path('data')
for domain in ['imagenet-c', 'ldct', 'dibco', 'fmd']:
    path = DATA_ROOT / domain
    status = "✓" if path.exists() and any(path.iterdir()) else "✗"
    print(f"{domain}: {status}")
```

DataLoader 테스트:

```python
import sys
sys.path.insert(0, 'src')
from data.datasets import get_dataset

for domain in ['imagenet-c', 'ldct', 'dibco', 'fmd']:
    root = f'data/{domain}'
    split = 'test' if domain == 'ldct' else 'train'
    ds = get_dataset(domain, root, split=split)
    print(f"{domain}: {len(ds)} samples")
```
