# Stage 5 RunPod Commands (Balance)

Run from your Stage 4 best checkpoint.

```bash
cd /workspace/DCT-GAN-Mobile-mobilelab
git checkout mobile-optimization-lab
git pull origin mobile-optimization-lab

# Confirm current baseline after Stage 4
python evaluate_mobile_recovery.py \
  --checkpoint checkpoints/best_model.pth \
  --config configs/mobile_stage4_strong_decoder.yaml \
  --dataset-path /workspace/DIV2K_prepared \
  --samples 120

# Stage 5 training (balance visual vs recovery)
python train.py \
  --config configs/mobile_stage5_balance.yaml \
  --dataset imagenet \
  --dataset-path /workspace/DIV2K_prepared \
  --resume checkpoints/best_model.pth

# Evaluate Stage 5 result
python evaluate_mobile_recovery.py \
  --checkpoint checkpoints/best_model.pth \
  --config configs/mobile_stage5_balance.yaml \
  --dataset-path /workspace/DIV2K_prepared \
  --samples 120
```

## Success criteria

- Visual PSNR > 16.3 dB
- Recovery PSNR >= 15.0 dB

If visual improves and recovery stays >= 15.0 dB, keep this line.
