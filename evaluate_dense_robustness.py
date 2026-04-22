"""
Evaluate Dense Robustness

Evaluacion de robustez para modelos Dense (Exp 18+) frente a:
- Compresion JPEG
- Rotaciones pequenas
- Traslaciones pequenas
- Escalado pequeno

Tambien soporta una compensacion "oracle" opcional para ataques geometricos,
aplicando la transformacion inversa conocida antes de decodificar. Esto sirve
como cota superior de un futuro modulo de estimacion geometrica.

Uso:
  python evaluate_dense_robustness.py \
    --checkpoint checkpoints/exp19_quality_base/best_model.pth \
    --val_dir /workspace/DIV2K_prepared/val/images \
    --samples 120 \
    --oracle_compensation
"""

import argparse
import io
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.models.dense_decoder import DenseDecoder, DenseDecoderLarge, DenseDecoderWithSkip
from src.models.dense_encoder import DenseEncoder, DenseEncoderLarge


def psnr_01(a: torch.Tensor, b: torch.Tensor) -> float:
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


def tensor_01_to_pil(image_01: torch.Tensor) -> Image.Image:
    tensor = image_01.detach().cpu().clamp(0.0, 1.0).squeeze(0)
    return TF.to_pil_image(tensor)


def pil_to_tensor_01(image: Image.Image) -> torch.Tensor:
    return TF.to_tensor(image).unsqueeze(0)


def apply_jpeg_attack(image_01: torch.Tensor, quality: int) -> torch.Tensor:
    pil_image = tensor_01_to_pil(image_01)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    attacked = Image.open(buffer).convert("RGB")
    return pil_to_tensor_01(attacked)


def apply_affine_attack(
    image_01: torch.Tensor,
    angle: float = 0.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
    scale: float = 1.0,
):
    attacked = TF.affine(
        image_01.squeeze(0),
        angle=angle,
        translate=[int(round(translate_x)), int(round(translate_y))],
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.5,
    )
    return attacked.unsqueeze(0)


def apply_inverse_affine(
    image_01: torch.Tensor,
    angle: float = 0.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
    scale: float = 1.0,
):
    inverse_scale = 1.0 / scale if abs(scale) > 1e-8 else 1.0
    restored = TF.affine(
        image_01.squeeze(0),
        angle=-angle,
        translate=[int(round(-translate_x)), int(round(-translate_y))],
        scale=inverse_scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.5,
    )
    return restored.unsqueeze(0)


def tensor_to_gray_numpy(image_01: torch.Tensor) -> np.ndarray:
    image_np = image_01.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    image_np = np.clip(image_np, 0.0, 1.0).astype(np.float32)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)


def numpy_to_tensor_01(image_np: np.ndarray) -> torch.Tensor:
    image_np = np.clip(image_np, 0.0, 1.0).astype(np.float32)
    if image_np.ndim == 2:
        image_np = np.expand_dims(image_np, axis=-1)
    image_np = np.transpose(image_np, (2, 0, 1))
    return torch.from_numpy(image_np).unsqueeze(0)


def estimate_and_compensate_affine(reference_01: torch.Tensor, attacked_01: torch.Tensor) -> torch.Tensor:
    reference_np = reference_01.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    attacked_np = attacked_01.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    reference_np = np.clip(reference_np, 0.0, 1.0).astype(np.float32)
    attacked_np = np.clip(attacked_np, 0.0, 1.0).astype(np.float32)

    reference_gray = cv2.cvtColor(reference_np, cv2.COLOR_RGB2GRAY)
    attacked_gray = cv2.cvtColor(attacked_np, cv2.COLOR_RGB2GRAY)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        100,
        1e-6,
    )

    try:
        _, warp_matrix = cv2.findTransformECC(
            reference_gray,
            attacked_gray,
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5,
        )
    except cv2.error:
        return attacked_01

    height, width = reference_gray.shape
    compensated = cv2.warpAffine(
        attacked_np,
        warp_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.5, 0.5, 0.5),
    )
    return numpy_to_tensor_01(compensated)


def parse_int_list(raw_value: str):
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_float_list(raw_value: str):
    return [float(item.strip()) for item in raw_value.split(",") if item.strip()]


def decode_secret(decoder, stego_01: torch.Tensor, device: torch.device) -> torch.Tensor:
    stego_11 = stego_01.to(device) * 2.0 - 1.0
    return decoder(stego_11)


def evaluate_attack_set(name, samples, decoder, device):
    psnr_values = []
    for sample in samples:
        recovered = decode_secret(decoder, sample["attacked"], device)
        psnr_values.append(psnr_01(sample["secret_01"].to(device), recovered))
    return {
        "attack": name,
        "recovery_psnr": sum(psnr_values) / len(psnr_values),
    }


def evaluate_compensated_set(name, samples, decoder, device):
    psnr_values = []
    for sample in samples:
        compensated = apply_inverse_affine(
            sample["attacked"],
            angle=sample["angle"],
            translate_x=sample["translate_x"],
            translate_y=sample["translate_y"],
            scale=sample["scale"],
        )
        recovered = decode_secret(decoder, compensated, device)
        psnr_values.append(psnr_01(sample["secret_01"].to(device), recovered))
    return {
        "attack": name,
        "recovery_psnr": sum(psnr_values) / len(psnr_values),
    }


def evaluate_auto_compensated_set(name, samples, decoder, device):
    psnr_values = []
    for sample in samples:
        compensated = estimate_and_compensate_affine(sample["reference_01"], sample["attacked"])
        recovered = decode_secret(decoder, compensated, device)
        psnr_values.append(psnr_01(sample["secret_01"].to(device), recovered))
    return {
        "attack": name,
        "recovery_psnr": sum(psnr_values) / len(psnr_values),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dense robustness against JPEG and geometric attacks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to dense checkpoint")
    parser.add_argument("--val_dir", type=str, required=True, help="Validation images directory")
    parser.add_argument("--samples", type=int, default=100, help="Number of pairs to evaluate")
    parser.add_argument("--image_size", type=int, default=256, help="Resize size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--jpeg_qualities", type=str, default="95,75,50", help="Comma-separated JPEG qualities")
    parser.add_argument("--rotation_angles", type=str, default="5,-5,10,-10", help="Comma-separated angles in degrees")
    parser.add_argument("--translations", type=str, default="4:0,-4:0,0:4,0:-4", help="Comma-separated tx:ty pixel shifts")
    parser.add_argument("--scales", type=str, default="0.95,1.05", help="Comma-separated scaling factors")
    parser.add_argument("--auto_compensation", action="store_true", help="Estimate and compensate small rotation/translation automatically")
    parser.add_argument("--oracle_compensation", action="store_true", help="Apply inverse geometric transform before decoding")
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
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    n = min(args.samples, len(images))
    jpeg_qualities = parse_int_list(args.jpeg_qualities)
    rotation_angles = parse_float_list(args.rotation_angles)
    translation_specs = [item.strip() for item in args.translations.split(",") if item.strip()]
    scale_values = parse_float_list(args.scales)

    baseline_samples = []
    jpeg_samples = {quality: [] for quality in jpeg_qualities}
    rotation_samples = {angle: [] for angle in rotation_angles}
    translation_samples = {spec: [] for spec in translation_specs}
    scale_samples = {scale: [] for scale in scale_values}

    print("=" * 60)
    print("EVALUACION DENSE ROBUSTNESS")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Modelo: encoder={encoder_type}, decoder={decoder_type}, hidden={hidden_size}")
    print(f"Val dir: {val_dir}")
    print(f"Samples: {n}")
    print(f"Auto compensation: {'ON' if args.auto_compensation else 'OFF'}")
    print(f"Oracle compensation: {'ON' if args.oracle_compensation else 'OFF'}")

    with torch.no_grad():
        for i in range(n):
            cover = tf(Image.open(images[i]).convert("RGB")).unsqueeze(0).to(device)

            j = random.randrange(len(images))
            while j == i and len(images) > 1:
                j = random.randrange(len(images))
            secret = tf(Image.open(images[j]).convert("RGB")).unsqueeze(0).to(device)

            stego_11 = encoder(cover, secret)
            stego_01 = ((stego_11 + 1.0) / 2.0).detach().cpu()
            secret_01 = ((secret + 1.0) / 2.0).detach().cpu()

            baseline_samples.append({
                "attacked": stego_01,
                "reference_01": stego_01,
                "secret_01": secret_01,
            })

            for quality in jpeg_qualities:
                jpeg_samples[quality].append({
                    "attacked": apply_jpeg_attack(stego_01, quality),
                    "reference_01": stego_01,
                    "secret_01": secret_01,
                })

            for angle in rotation_angles:
                attacked = apply_affine_attack(stego_01, angle=angle)
                rotation_samples[angle].append({
                    "attacked": attacked,
                    "reference_01": stego_01,
                    "secret_01": secret_01,
                    "angle": angle,
                    "translate_x": 0.0,
                    "translate_y": 0.0,
                    "scale": 1.0,
                })

            for spec in translation_specs:
                tx_raw, ty_raw = spec.split(":", maxsplit=1)
                tx = float(tx_raw)
                ty = float(ty_raw)
                attacked = apply_affine_attack(stego_01, translate_x=tx, translate_y=ty)
                translation_samples[spec].append({
                    "attacked": attacked,
                    "reference_01": stego_01,
                    "secret_01": secret_01,
                    "angle": 0.0,
                    "translate_x": tx,
                    "translate_y": ty,
                    "scale": 1.0,
                })

            for scale in scale_values:
                attacked = apply_affine_attack(stego_01, scale=scale)
                scale_samples[scale].append({
                    "attacked": attacked,
                    "reference_01": stego_01,
                    "secret_01": secret_01,
                    "angle": 0.0,
                    "translate_x": 0.0,
                    "translate_y": 0.0,
                    "scale": scale,
                })

            if (i + 1) % 10 == 0:
                print(f"  Procesadas {i + 1}/{n}")

    results = [evaluate_attack_set("baseline", baseline_samples, decoder, device)]

    for quality, samples in jpeg_samples.items():
        results.append(evaluate_attack_set(f"jpeg_q{quality}", samples, decoder, device))

    for angle, samples in rotation_samples.items():
        results.append(evaluate_attack_set(f"rotate_{angle:+.0f}", samples, decoder, device))
        if args.auto_compensation:
            results.append(evaluate_auto_compensated_set(f"rotate_{angle:+.0f}_auto", samples, decoder, device))
        if args.oracle_compensation:
            results.append(evaluate_compensated_set(f"rotate_{angle:+.0f}_oracle", samples, decoder, device))

    for spec, samples in translation_samples.items():
        results.append(evaluate_attack_set(f"translate_{spec}", samples, decoder, device))
        if args.auto_compensation:
            results.append(evaluate_auto_compensated_set(f"translate_{spec}_auto", samples, decoder, device))
        if args.oracle_compensation:
            results.append(evaluate_compensated_set(f"translate_{spec}_oracle", samples, decoder, device))

    for scale, samples in scale_samples.items():
        results.append(evaluate_attack_set(f"scale_{scale:.2f}", samples, decoder, device))
        if args.oracle_compensation:
            results.append(evaluate_compensated_set(f"scale_{scale:.2f}_oracle", samples, decoder, device))

    print("\n" + "=" * 60)
    print("RESULTADOS ROBUSTEZ")
    print("=" * 60)
    print(f"{'Ataque':<28} {'Rec PSNR (dB)':>14}")
    print("-" * 60)
    for item in results:
        print(f"{item['attack']:<28} {item['recovery_psnr']:>14.2f}")


if __name__ == "__main__":
    main()