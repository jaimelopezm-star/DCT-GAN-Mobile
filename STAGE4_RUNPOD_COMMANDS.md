# Stage 4 RunPod Commands (Strong Decoder Pivot)

This stage performs an architecture pivot using partial checkpoint loading.

```bash
cd /workspace/DCT-GAN-Mobile-mobilelab
git checkout mobile-optimization-lab
git pull origin mobile-optimization-lab

# Baseline eval (before Stage 4)
python evaluate_mobile_recovery.py \
  --checkpoint checkpoints/best_model.pth \
  --config configs/mobile_finetune_stage3b_visual.yaml \
  --dataset-path /workspace/DIV2K_prepared \
  --samples 120

# Stage 4 training (strong decoder, recovery-first)
python train.py \
  --config configs/mobile_stage4_strong_decoder.yaml \
  --dataset imagenet \
  --dataset-path /workspace/DIV2K_prepared \
  --resume checkpoints/best_model.pth

# Eval after Stage 4
python evaluate_mobile_recovery.py \
  --checkpoint checkpoints/best_model.pth \
  --config configs/mobile_stage4_strong_decoder.yaml \
  --dataset-path /workspace/DIV2K_prepared \
  --samples 120
```

## Expected behavior

- Training log should show:
  - `Partial resume enabled`
  - `Reset epoch and best_psnr for partial resume`
- This is normal and means encoder/discriminator weights were transferred while
  decoder-specific layers were initialized for the new architecture.
