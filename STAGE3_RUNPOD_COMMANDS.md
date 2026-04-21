# Stage 3 RunPod Commands

Run these commands in order inside RunPod.

```bash
cd /workspace/DCT-GAN-Mobile-mobilelab
git checkout mobile-optimization-lab
git pull origin mobile-optimization-lab

# Baseline eval before Stage 3
python evaluate_mobile_recovery.py \
  --checkpoint checkpoints/best_model.pth \
  --config configs/mobile_finetune_stage2b.yaml \
  --dataset-path /workspace/DIV2K_prepared \
  --samples 120

# Stage 3A: recovery-focused
python train.py \
  --config configs/mobile_finetune_stage3a_recovery.yaml \
  --dataset imagenet \
  --dataset-path /workspace/DIV2K_prepared \
  --resume checkpoints/best_model.pth

# Eval after Stage 3A
python evaluate_mobile_recovery.py \
  --checkpoint checkpoints/best_model.pth \
  --config configs/mobile_finetune_stage3a_recovery.yaml \
  --dataset-path /workspace/DIV2K_prepared \
  --samples 120

# Stage 3B: visual refinement
python train.py \
  --config configs/mobile_finetune_stage3b_visual.yaml \
  --dataset imagenet \
  --dataset-path /workspace/DIV2K_prepared \
  --resume checkpoints/best_model.pth

# Final eval after Stage 3B
python evaluate_mobile_recovery.py \
  --checkpoint checkpoints/best_model.pth \
  --config configs/mobile_finetune_stage3b_visual.yaml \
  --dataset-path /workspace/DIV2K_prepared \
  --samples 120
```

## Quick decision rule

- Continue if both conditions hold:
  - visual PSNR improves by >= 0.3 dB versus Stage 2B baseline
  - recovery PSNR does not drop versus Stage 2B baseline
- Stop and pivot architecture if one condition fails.
