# Notebooks

연구자를 위한 대화형 노트북입니다. 파이프라인 코드를 수정하지 않고 데이터 탐색, 시각화, 모델 테스트를 수행할 수 있습니다.

## 노트북 목록

| 노트북 | 설명 | 데이터 필요 |
|--------|------|-------------|
| `01_data_exploration.ipynb` | 데이터셋 탐색 및 샘플 시각화 | O |
| `02_model_visualization.ipynb` | 학습된 모델 결과 분석 | O |
| `03_quick_demo.ipynb` | 합성 이미지로 빠른 테스트 | X |

## 사용법

### 1. Jupyter Notebook 실행

```bash
# 가상환경 활성화 후
cd notebooks
jupyter notebook
```

또는 VS Code에서 직접 열기

### 2. 노트북별 가이드

#### 01_data_exploration.ipynb
- **목적**: 각 도메인(ImageNet-C, LDCT, DIBCO, FMD) 데이터 확인
- **수정 필요**: `DATA_ROOT` 경로를 실제 데이터 위치로 변경
- **기능**:
  - 샘플 이미지 그리드 표시
  - 도메인 간 비교
  - 히스토그램 분석

#### 02_model_visualization.ipynb
- **목적**: 학습된 모델의 복원 결과 시각화
- **수정 필요**:
  - `CHECKPOINT_PATH`: 학습된 모델 체크포인트 경로
  - `DATA_ROOT`: 데이터 경로
- **기능**:
  - 복원 결과 (Degraded → Restored → GT)
  - t-SNE 시각화 (열화 표현 z_d)
  - Cross-domain transfer 결과
  - PSNR/SSIM 메트릭

#### 03_quick_demo.ipynb
- **목적**: 데이터 없이 모델 빠르게 테스트
- **수정 필요**: 없음 (선택적으로 `IMAGE_PATH` 설정)
- **기능**:
  - 합성 열화 이미지 생성 (노이즈, 블러)
  - 복원 결과 확인
  - 사용자 이미지 테스트
  - 모델 구조 확인

## examples 폴더

`notebooks/examples/` 폴더에 테스트용 이미지를 넣어두면 노트북에서 쉽게 불러올 수 있습니다.

```
notebooks/
├── examples/
│   ├── sample_noisy.png
│   ├── sample_blurry.jpg
│   └── ...
├── 01_data_exploration.ipynb
├── 02_model_visualization.ipynb
├── 03_quick_demo.ipynb
└── README.md
```

## 의존성

노트북 실행에 필요한 추가 패키지:

```bash
pip install jupyter ipykernel
```

## 문제 해결

### Import 에러
노트북 첫 셀에서 `sys.path.insert(0, '..')` 가 제대로 작동하지 않으면:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
```

### 메모리 부족
큰 이미지나 많은 샘플 시 메모리 문제 발생 시:
- `n_samples` 값을 줄이기
- 이미지 크기를 128x128로 줄이기
