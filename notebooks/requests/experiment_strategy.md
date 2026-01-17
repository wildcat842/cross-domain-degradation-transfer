## 이름 규칙 : 모델, 학습세팅, 데이터셋
## denoiser - baseline 훈련 명령어 --> dncnn_noiseonly_imagenet 에 저장
python scripts/train.py --config  configs/dcnn_noiseonly.yaml --model denoiser

## CDDT-mulidomain 훈련 명령어  --> cddt_multidomain_imagenetNBC 에 저장
python scripts/train.py --config  configs/cddt_multidomain.yaml

## CDDT noise only 훈련 명령어  --> cddt_noiseonly_imagenetN 에 저장
python scripts/train.py --config  configs/cddt_noiseonly.yaml


# Noise → Blur cross-corruption transfer  (전이 실험) --> crrupption_transfer 에 저장
# (source, target 주는 순간 config 'domains' 는 완전히 무시됨) -
python scripts/train.py --config configs/corruption_transfer.yaml \
      --source_domain imagenet-noise --target_domain imagenet-blur

  # Noise → LDCT cross-domain transfer --> cross_domain_transfer 에 저장
python scripts/train.py --config configs/cross_domain_transfer.yaml \
      --source_domain imagenet-noise --target_domain ldct

