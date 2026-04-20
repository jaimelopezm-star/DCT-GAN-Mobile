"""
Evaluate Dense Recovery

Evaluacion separada para modelos Dense (Exp 18+) usando checkpoints generados por
`train_dense.py`.

Metricas:
- PSNR visual (cover vs stego)
- PSNR recovery (secret vs recovered)

Uso:
  python evaluate_dense_recovery.py \
    --checkpoint checkpoints/exp18_finetune_recovery/best_model.pth \
    --val_dir /workspace/DIV2K_prepared/val/images \
    --samples 100
"""

import argparse
import math
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.models.dense_encoder import DenseEncoder, DenseEncoderLarge
from src.models.dense_decoder import DenseDecoder, DenseDecoderLarge, DenseDecoderWithSkip


def psnr_01(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR para tensores en rango [0, 1]."""
    mse = torch.mean((a - b) ** 2).item()
    if mse < 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def collect_images(image_dir: Path):
    images = []
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.PNG", "*.JPG", "*.JPEG", "*.BMP")
    for ext in exts:
        images.extend(image_dir.glob(ext))
    return sorted(images)


def build_models_from_checkpoint(checkpoint, device):
    cfg = checkpoint.get("config", {})
    encoder_type = cfg.get("encoder_type", "dense")
    decoder_type = cfg.get("decoder_type", "dense")
    hidden_size = int(cfg.get("hidden_size", 64))

    if encoder_type == "dense":
        encoder = DenseEncoder(data_depth=3, hidden_size=hidden_size)
    else:
        encoder = DenseEncoderLarge(data_depth=3, hidden_size=hidden_size)

    if decoder_type == "dense":
        decoder = DenseDecoder(data_depth=3, hidden_size=hidden_size)
    elif decoder_type == "dense_large":
        decoder = DenseDecoderLarge(data_depth=3, hidden_size=hidden_size)
    else:
        decoder = DenseDecoderWithSkip(data_depth=3, hidden_size=hidden_size)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.to(device).eval()
    decoder.to(device).eval()

    return encoder, decoder, hidden_size, encoder_type, decoder_type


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dense steganography recovery")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to dense checkpoint")
    parser.add_argument("--val_dir", type=str, required=True, help="Validation images directory")
    parser.add_argument("--samples", type=int, default=100, help="Number of pairs to evaluate")
    parser.add_argument("--image_size", type=int, default=256, help="Resize size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    val_dir = Path(args.val_dir)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Directorio de validacion no encontrado: {val_dir}")

    images = collect_images(val_dir)
    if len(images) < 2:
        raise ValueError(f"Se necesitan al menos 2 imagenes en {val_dir}")

    checkpoint = torch.load(str(ckpt_path), map_location=device)
    encoder, decoder, hidden_size, encoder_type, decoder_type = build_models_from_checkpoint(checkpoint, device)

    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
    ])

    n = min(args.samples, len(images))
    psnr_visual = []
    psnr_recovery = []

    print("=" * 60)
    print("EVALUACION DENSE RECOVERY")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Modelo: encoder={encoder_type}, decoder={decoder_type}, hidden={hidden_size}")
    print(f"Val dir: {val_dir}")
    print(f"Samples: {n}")

    with torch.no_grad():
        for i in range(n):
            cover = tf(Image.open(images[i]).convert("RGB")).unsqueeze(0).to(device)

            j = random.randrange(len(images))
            while j == i and len(images) > 1:
                j = random.randrange(len(images))
            secret = tf(Image.open(images[j]).convert("RGB")).unsqueeze(0).to(device)

            stego = encoder(cover, secret)      # [-1, 1]
            recovered = decoder(stego)          # [0, 1]

            cover_01 = (cover + 1.0) / 2.0
            stego_01 = (stego + 1.0) / 2.0
            secret_01 = (secret + 1.0) / 2.0

            psnr_visual.append(psnr_01(cover_01, stego_01))
            psnr_recovery.append(psnr_01(secret_01, recovered))

            if (i + 1) % 10 == 0:
                print(f"  Procesadas {i + 1}/{n}")

    avg_visual = sum(psnr_visual) / len(psnr_visual)
    avg_recovery = sum(psnr_recovery) / len(psnr_recovery)

    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"PSNR visual (cover vs stego): {avg_visual:.2f} dB")
    print(f"PSNR recovery (secret vs recovered): {avg_recovery:.2f} dB")

    if avg_recovery >= 20.0:
        print("✅ Recuperacion: EXITOSA (>= 20 dB)")
    else:
        print("⚠️ Recuperacion: EN MEJORA (< 20 dB)")


if __name__ == "__main__":
    main()
