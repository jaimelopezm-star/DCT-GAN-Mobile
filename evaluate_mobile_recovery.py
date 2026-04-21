"""
Evaluate Mobile Recovery with real validation images.

Metrics:
- PSNR visual (cover vs stego)
- SSIM visual (cover vs stego)
- PSNR recovery (secret vs recovered)
- SSIM recovery (secret vs recovered)

Usage:
  python evaluate_mobile_recovery.py \
    --checkpoint checkpoints/best_model.pth \
    --config configs/mobile_finetune_stage2b.yaml \
    --dataset-path /workspace/DIV2K_prepared \
    --samples 120
"""

import argparse
import random
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.gan import DCTGAN
from training.losses import calculate_psnr
from training.metrics import calculate_ssim


def collect_images(image_dir: Path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.PNG", "*.JPG", "*.JPEG", "*.BMP")
    files = []
    for ext in exts:
        files.extend(image_dir.glob(ext))
    return sorted(files)


def resolve_val_dir(dataset_path: Path) -> Path:
    candidates = [
        dataset_path / "val" / "images",
        dataset_path / "val",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No validation directory found under {dataset_path}. Expected val/images or val")


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    model = DCTGAN(
        encoder_config=model_cfg.get("encoder", {}),
        decoder_config=model_cfg.get("decoder", {}),
        discriminator_config=model_cfg.get("discriminator", {}),
    )

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate mobile checkpoint on real data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--config", type=str, required=True, help="Config path used by model")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset root path")
    parser.add_argument("--samples", type=int, default=120, help="Number of random pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    dataset_path = Path(args.dataset_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    val_dir = resolve_val_dir(dataset_path)
    images = collect_images(val_dir)
    if len(images) < 2:
        raise ValueError(f"Need at least 2 images in {val_dir}")

    model = load_model(config_path, checkpoint_path, device)

    image_size = 256
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        image_size = int(cfg.get("data", {}).get("image_size", 256))

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0, 1], same as train.py real-dataset path
    ])

    num_samples = min(args.samples, len(images))

    psnr_visual = []
    ssim_visual = []
    psnr_recovery = []
    ssim_recovery = []

    print("=" * 68)
    print("MOBILE REAL-DATA EVALUATION")
    print("=" * 68)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Validation directory: {val_dir}")
    print(f"Samples: {num_samples}")

    with torch.no_grad():
        for i in range(num_samples):
            cover_idx = i
            secret_idx = random.randrange(len(images))
            while secret_idx == cover_idx and len(images) > 1:
                secret_idx = random.randrange(len(images))

            cover = tf(Image.open(images[cover_idx]).convert("RGB")).unsqueeze(0).to(device)
            secret = tf(Image.open(images[secret_idx]).convert("RGB")).unsqueeze(0).to(device)

            stego, recovered = model(cover, secret, mode="full")

            p_vis = calculate_psnr(cover.float(), stego.float(), max_val=1.0).item()
            s_vis = float(calculate_ssim(cover.float(), stego.float()))
            p_rec = calculate_psnr(secret.float(), recovered.float(), max_val=1.0).item()
            s_rec = float(calculate_ssim(secret.float(), recovered.float()))

            psnr_visual.append(p_vis)
            ssim_visual.append(s_vis)
            psnr_recovery.append(p_rec)
            ssim_recovery.append(s_rec)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{num_samples}")

    avg_psnr_visual = sum(psnr_visual) / len(psnr_visual)
    avg_ssim_visual = sum(ssim_visual) / len(ssim_visual)
    avg_psnr_recovery = sum(psnr_recovery) / len(psnr_recovery)
    avg_ssim_recovery = sum(ssim_recovery) / len(ssim_recovery)

    print("\n" + "=" * 68)
    print("RESULTS")
    print("=" * 68)
    print(f"Visual PSNR (cover vs stego):   {avg_psnr_visual:.2f} dB")
    print(f"Visual SSIM (cover vs stego):   {avg_ssim_visual:.4f}")
    print(f"Recovery PSNR (secret vs rec.): {avg_psnr_recovery:.2f} dB")
    print(f"Recovery SSIM (secret vs rec.): {avg_ssim_recovery:.4f}")


if __name__ == "__main__":
    main()
