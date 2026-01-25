## 이 코드는 이렇게 실행되었다. 
python ./scripts/evaluate.py --checkpoint experiments/cddt_multidomain/cddt_multidomain_20260115_170401/checkpoint_0077_best.pth --config ./configs/evaluate.yaml --n_shots 10

## 결과는 다음과 같다. 

Using device: cuda
Loading model from: experiments/cddt_multidomain/cddt_multidomain_20260115_170401/checkpoint_0077_best.pth

==================================================
Cross-Domain Transfer Evaluation Matrix
==================================================

Evaluating: imagenet -> ldct
Evaluating imagenet->ldct: 100%|███████████████████████| 112/112 [00:03<00:00, 30.75it/s]
  PSNR: 25.28, SSIM: 0.5864

Evaluating: imagenet -> dibco
Evaluating imagenet->dibco: 100%|██████████████████████████| 1/1 [00:01<00:00,  1.59s/it]
  PSNR: 5.76, SSIM: 0.2601

Evaluating: imagenet -> fmd
Evaluating imagenet->fmd: 100%|████████████████████████| 188/188 [00:06<00:00, 29.04it/s]
  PSNR: 29.70, SSIM: 0.7703

Evaluating: ldct -> imagenet
Evaluating ldct->imagenet: 100%|█████████████████████| 8125/8125 [02:42<00:00, 50.04it/s]
  PSNR: 26.06, SSIM: 0.6780

Evaluating: ldct -> dibco
Evaluating ldct->dibco: 100%|██████████████████████████████| 1/1 [00:01<00:00,  1.52s/it]
  PSNR: 5.78, SSIM: 0.2529

Evaluating: ldct -> fmd
Evaluating ldct->fmd: 100%|████████████████████████████| 188/188 [00:06<00:00, 28.80it/s]
  PSNR: 29.73, SSIM: 0.7734

Evaluating: dibco -> imagenet
Evaluating dibco->imagenet: 100%|████████████████████| 8125/8125 [02:42<00:00, 49.99it/s]
  PSNR: 26.05, SSIM: 0.6783

Evaluating: dibco -> ldct
Evaluating dibco->ldct: 100%|██████████████████████████| 112/112 [00:03<00:00, 30.71it/s]
  PSNR: 25.27, SSIM: 0.5875

Evaluating: dibco -> fmd
Evaluating dibco->fmd: 100%|███████████████████████████| 188/188 [00:06<00:00, 29.63it/s]
  PSNR: 29.71, SSIM: 0.7713

Evaluating: fmd -> imagenet
Evaluating fmd->imagenet: 100%|██████████████████████| 8125/8125 [02:42<00:00, 49.99it/s]
  PSNR: 26.06, SSIM: 0.6786

Evaluating: fmd -> ldct
Evaluating fmd->ldct: 100%|████████████████████████████| 112/112 [00:03<00:00, 33.33it/s]
  PSNR: 25.27, SSIM: 0.5871

Evaluating: fmd -> dibco
Evaluating fmd->dibco: 100%|███████████████████████████████| 1/1 [00:01<00:00,  1.67s/it]
  PSNR: 5.80, SSIM: 0.2578

PSNR Matrix (Source -> Target):
target       dibco        fmd   imagenet       ldct
source                                             
dibco          NaN  29.713360  26.054975  25.267941
fmd       5.803712        NaN  26.056232  25.266576
imagenet  5.764524  29.699382        NaN  25.276719
ldct      5.775225  29.732170  26.056018        NaN

Average PSNR: 21.71
Average SSIM: 0.5735

Generating t-SNE visualization...
[imagenet] Loaded 130000 samples (batch_size=16)
Evaluating imagenet: 100%|███████████████████████████| 8125/8125 [02:43<00:00, 49.80it/s]
[ldct] Loaded 1792 samples (batch_size=16)
Evaluating ldct: 100%|█████████████████████████████████| 112/112 [00:03<00:00, 28.98it/s]
[dibco] Loaded 9 samples (batch_size=9)
Evaluating dibco: 100%|████████████████████████████████████| 1/1 [00:01<00:00,  1.37s/it]
[fmd] Loaded 3000 samples (batch_size=16)
Evaluating fmd: 100%|██████████████████████████████████| 188/188 [00:06<00:00, 29.58it/s]
t-SNE saved to: results/imagenetmult_all_evaluation/tsne_degradation_space.png

Results saved to: results/imagenetmult_all_evaluation