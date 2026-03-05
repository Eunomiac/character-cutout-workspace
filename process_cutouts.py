"""
Character Cutout Workspace - V:tM NPC Image Processing Pipeline

Processes images through: upscale -> segment -> cutout -> alpha boost -> downscale ->
smart crop -> color match -> vampiric correction.

Run from project root: python process_cutouts.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# Route model caches to workspace (saves SSD space; allows HDD storage)
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_CACHE = _SCRIPT_DIR / "model-cache"
_MODEL_CACHE.mkdir(exist_ok=True)
(_MODEL_CACHE / "rembg").mkdir(exist_ok=True)
(_MODEL_CACHE / "huggingface").mkdir(exist_ok=True)
os.environ.setdefault("U2NET_HOME", str(_MODEL_CACHE / "rembg"))
os.environ.setdefault("HF_HOME", str(_MODEL_CACHE / "huggingface"))

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

# === CONFIGURATION (edit these) ===
# Environment: CUDA available. Model-size options default to largest/best for quality.
INPUT_FOLDER = "input"
OUTPUT_BASE = "output"
MODELS = ["birefnet-general", "birefnet-portrait", "bria-rmbg", "sam2"]
INPUT_EXTENSIONS = (".png", ".tiff", ".tif")

# Target height for all final cutouts (width scales proportionally)
TARGET_HEIGHT = 3000

# Upscale to 1.5x target height before cutout (gives model room for removal)
UPSCALE_HEIGHT_FACTOR = 1.5  # working height = TARGET_HEIGHT * UPSCALE_HEIGHT_FACTOR

ENABLE_EXPOSURE_ADJUST = True
EXPOSURE_GAMMA = 0.55  # < 1 brightens segmentation input only; final cutout unchanged

# SAM 2 only: model size (tiny/small/base_plus/large). Use "large" for best quality with CUDA.
SAM2_MODEL_SIZE = "large"

# Alpha matting (rembg models only): refines edges for hair/wispy details.
ENABLE_ALPHA_MATTING = True
ALPHA_MATTING_FOREGROUND_THRESHOLD = 240
ALPHA_MATTING_BACKGROUND_THRESHOLD = 10
ALPHA_MATTING_ERODE_STRUCTURE_SIZE = 10

# Vampiric correction: skin-only color adjustment (paler, cooler).
ENABLE_VAMPIRIC_CORRECTION = True
VAMPIRIC_PALENESS = 0.85
VAMPIRIC_COOL_SHIFT = 0.08
VAMPIRIC_SATURATION = 0.9

# Batch color matching
ENABLE_COLOR_MATCHING = True
COLOR_MATCH_REFERENCE = "color-references/male-vampire-5.png"

# Inhuman/monster/masked: skip vampiric correction
INHUMAN_DETECTION = "auto"
INHUMAN_SKIN_THRESHOLD = 0.08
INHUMAN_FILENAME_SUFFIX = "-inhuman"

# Smart crop
ENABLE_SMART_CROP = True
CROP_ALPHA_THRESHOLD = 0.4
CROP_PADDING = 5

# Alpha boost
ENABLE_ALPHA_BOOST = True
ALPHA_BOOST_PASSES = 4

# Working height before cutout (1.5x target)
WORKING_HEIGHT = int(TARGET_HEIGHT * UPSCALE_HEIGHT_FACTOR)

# SAM 2 HuggingFace model IDs by size
SAM2_HF_IDS: dict[str, str] = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}


def adjust_exposure(img: PILImage, gamma: float) -> PILImage:
    """Apply gamma correction to brighten image for segmentation. Does not modify original."""
    from skimage.exposure import adjust_gamma

    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        rgb = arr[:, :, :3].astype(np.float64) / 255.0
        rgb = adjust_gamma(rgb, gamma)
        arr = np.copy(arr)
        arr[:, :, :3] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr)
    else:
        arr = arr[:, :, :3].astype(np.float64) / 255.0
    adjusted = adjust_gamma(arr, gamma)
    adjusted = (np.clip(adjusted, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(adjusted)


def upscale_image(img: PILImage, target_height: int) -> PILImage:
    """Upscale image so its height reaches target_height using Real-ESRGAN."""
    h, w = img.height, img.width
    if h >= target_height:
        return img
    scale = target_height / h
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    weights_dir = _MODEL_CACHE / "real-esrgan"
    weights_dir.mkdir(exist_ok=True)
    model_path = weights_dir / "RealESRGAN_x4plus.pth"
    if not model_path.exists():
        downloaded = load_file_from_url(
            url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            model_dir=str(weights_dir),
            progress=True,
        )
        model_path = Path(downloaded)
    upsampler = RealESRGANer(scale=4, model_path=str(model_path), model=model, half=True, gpu_id=0)
    img_arr = np.array(img.convert("RGB"))
    img_arr = img_arr[:, :, ::-1]  # RGB -> BGR for cv2-style
    outscale = min(4.0, max(2.0, scale))
    output, _ = upsampler.enhance(img_arr, outscale=outscale)
    if output.shape[0] < target_height:
        pil_out = Image.fromarray(output[:, :, ::-1].copy())
        new_w = int(w * target_height / h)
        pil_out = pil_out.resize((new_w, target_height), Image.Resampling.LANCZOS)
        return pil_out
    result = Image.fromarray(output[:, :, ::-1].copy())
    if result.height > target_height:
        ratio = target_height / result.height
        new_w = int(result.width * ratio)
        result = result.resize((new_w, target_height), Image.Resampling.LANCZOS)
    return result


def generate_mask(img: PILImage, model_name: str, sessions: dict[str, Any]) -> PILImage:
    """Generate foreground mask using rembg or sam2. Returns single-channel mask (L mode)."""
    if model_name == "sam2":
        return _generate_mask_sam2(img, sessions.get("sam2_predictor"))
    session = sessions.get(model_name)
    if session is None:
        raise ValueError(f"Unknown model: {model_name}")
    from rembg import remove

    result = remove(
        img,
        session=session,
        alpha_matting=ENABLE_ALPHA_MATTING,
        alpha_matting_foreground_threshold=ALPHA_MATTING_FOREGROUND_THRESHOLD,
        alpha_matting_background_threshold=ALPHA_MATTING_BACKGROUND_THRESHOLD,
        alpha_matting_erode_structure_size=ALPHA_MATTING_ERODE_STRUCTURE_SIZE,
    )
    if isinstance(result, Image.Image):
        if result.mode == "RGBA":
            return result.split()[-1]
        return result.convert("L")
    raise RuntimeError("rembg did not return a PIL Image")


def _generate_mask_sam2(img: PILImage, predictor: Any) -> PILImage:
    """Generate mask using SAM 2 with center-point prompt."""
    if predictor is None:
        raise RuntimeError("SAM 2 predictor not initialized")
    import torch

    img_np = np.array(img.convert("RGB"))
    h, w = img_np.shape[:2]
    center_x, center_y = w // 2, h // 2
    point_coords = np.array([[center_x, center_y]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int64)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(img_np)
        masks, iou_preds, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            return_logits=False,
        )
    best_idx = int(np.argmax(iou_preds))
    mask_np = (masks[best_idx] * 255).astype(np.uint8)
    return Image.fromarray(mask_np, mode="L")


def alpha_boost(img: PILImage) -> PILImage:
    """Apply alpha boost: new_alpha = 1 - (1 - alpha)^N."""
    if img.mode != "RGBA":
        return img
    arr = np.array(img)
    alpha = arr[:, :, 3].astype(np.float64) / 255.0
    alpha = 1.0 - (1.0 - alpha) ** ALPHA_BOOST_PASSES
    arr[:, :, 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def smart_crop(img: PILImage) -> PILImage:
    """Crop to content bounds (alpha >= threshold), add padding, re-scale to TARGET_HEIGHT."""
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] < 4:
        return img
    alpha = arr[:, :, 3].astype(np.float64) / 255.0
    thresh = CROP_ALPHA_THRESHOLD
    content = alpha >= thresh
    if not np.any(content):
        return img
    rows = np.any(content, axis=1)
    cols = np.any(content, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = CROP_PADDING
    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0], rmax + pad + 1)
    cmin = max(0, cmin - pad)
    cmax = min(arr.shape[1], cmax + pad + 1)
    cropped = arr[rmin:rmax, cmin:cmax]
    pil_crop = Image.fromarray(cropped)
    scale = TARGET_HEIGHT / pil_crop.height
    new_w = int(pil_crop.width * scale)
    return pil_crop.resize((new_w, TARGET_HEIGHT), Image.Resampling.LANCZOS)


def should_skip_vampiric(img: PILImage, filename: str) -> bool:
    """Return True if vampiric correction should be skipped (inhuman/masked/monster)."""
    if INHUMAN_DETECTION == "none":
        return False
    if INHUMAN_DETECTION == "suffix":
        return INHUMAN_FILENAME_SUFFIX in filename
    if INHUMAN_DETECTION == "auto":
        skin_ratio = _get_skin_coverage(img)
        return skin_ratio < INHUMAN_SKIN_THRESHOLD
    return False


def _get_skin_coverage(img: PILImage) -> float:
    """Return fraction of foreground pixels that are detected as skin (YCbCr)."""
    arr = np.array(img.convert("RGB"))
    if arr.ndim != 3:
        return 0.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    if img.mode == "RGBA":
        alpha = np.array(img.split()[-1]) / 255.0
    else:
        alpha = np.ones((arr.shape[0], arr.shape[1]))
    fg = alpha > 0.1
    if not np.any(fg):
        return 0.0
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.169 * r - 0.331 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.419 * g - 0.081 * b
    skin = (
        (y > 60) & (y < 255) &
        (cb > 77) & (cb < 127) &
        (cr > 133) & (cr < 173)
    )
    skin_fg = np.sum(skin & fg)
    total_fg = np.sum(fg)
    if total_fg == 0:
        return 0.0
    return float(skin_fg) / float(total_fg)


def apply_vampiric_correction(img: PILImage) -> PILImage:
    """Apply paler, cooler skin-only adjustment."""
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return img
    rgb = arr[:, :, :3].astype(np.float64)
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else np.ones((arr.shape[0], arr.shape[1]), dtype=np.uint8) * 255
    skin_mask = _get_skin_mask(rgb)
    from skimage.color import rgb2hsv, hsv2rgb

    hsv = rgb2hsv(rgb / 255.0)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    v_adj = v * VAMPIRIC_PALENESS
    h_adj = (h + VAMPIRIC_COOL_SHIFT) % 1.0  # Shift toward blue (cooler)
    s_adj = s * VAMPIRIC_SATURATION
    hsv_adj = np.stack([h_adj, s_adj, v_adj], axis=-1)
    rgb_adj = (hsv2rgb(hsv_adj) * 255).clip(0, 255).astype(np.uint8)
    blend = skin_mask[:, :, np.newaxis].astype(np.float64) / 255.0
    result_rgb = (rgb * (1 - blend) + rgb_adj * blend).astype(np.uint8)
    if arr.shape[2] == 4:
        result = np.dstack([result_rgb, alpha])
    else:
        result = result_rgb
    return Image.fromarray(result)


def _get_skin_mask(rgb: np.ndarray) -> np.ndarray:
    """YCbCr-based skin mask, 0-255."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.169 * r - 0.331 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.419 * g - 0.081 * b
    skin = (
        (y > 60) & (y < 255) &
        (cb > 77) & (cb < 127) &
        (cr > 133) & (cr < 173)
    )
    from skimage.filters import gaussian

    smooth = gaussian(skin.astype(np.float32), sigma=2)
    return (np.clip(smooth, 0, 1) * 255).astype(np.uint8)


def color_match_batch(paths: list[Path], reference_path: Path) -> None:
    """Match LAB stats of all images to reference; overwrite in place."""
    from skimage.color import lab2rgb, rgb2lab

    ref_img = np.array(Image.open(reference_path).convert("RGB"))
    ref_lab = rgb2lab(ref_img)
    ref_mean = np.array([np.mean(ref_lab[:, :, i]) for i in range(3)])
    ref_std = np.array([np.std(ref_lab[:, :, i]) for i in range(3)])
    ref_std = np.where(ref_std < 1e-6, 1.0, ref_std)
    for p in paths:
        img = Image.open(p)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        arr = np.array(img)
        rgb = arr[:, :, :3].astype(np.float64) / 255.0
        lab = rgb2lab(rgb)
        for i in range(3):
            mean_i = np.mean(lab[:, :, i])
            std_i = np.std(lab[:, :, i])
            if std_i < 1e-6:
                std_i = 1.0
            lab[:, :, i] = (lab[:, :, i] - mean_i) / std_i * ref_std[i] + ref_mean[i]
        rgb_out = (np.clip(lab2rgb(lab), 0, 1) * 255).astype(np.uint8)
        arr[:, :, :3] = rgb_out
        Image.fromarray(arr).save(p)


def process_image(
    img_path: Path,
    model_name: str,
    sessions: dict[str, Any],
    base_dir: Path,
) -> Optional[Path]:
    """Full pipeline for one image and one model. Returns output path or None on error."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  Error loading {img_path}: {e}")
        return None
    orig_size = img.size
    working_h = WORKING_HEIGHT
    if img.height > working_h:
        scale = working_h / img.height
        img = img.resize((int(img.width * scale), working_h), Image.Resampling.LANCZOS)
    elif img.height < working_h:
        img = upscale_image(img, working_h)
    seg_input = adjust_exposure(img, EXPOSURE_GAMMA) if ENABLE_EXPOSURE_ADJUST else img
    mask = generate_mask(seg_input, model_name, sessions)
    mask = mask.resize(img.size, Image.Resampling.LANCZOS)
    composite = Image.new("RGBA", img.size, (0, 0, 0, 0))
    composite.paste(img, (0, 0))
    composite.putalpha(mask)
    if ENABLE_ALPHA_BOOST:
        composite = alpha_boost(composite)
    scale = TARGET_HEIGHT / composite.height
    composite = composite.resize(
        (int(composite.width * scale), TARGET_HEIGHT),
        Image.Resampling.LANCZOS,
    )
    if ENABLE_SMART_CROP:
        composite = smart_crop(composite)
    out_dir = base_dir / OUTPUT_BASE / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / img_path.name
    if out_path.suffix.lower() not in (".png", ".tiff", ".tif"):
        out_path = out_path.with_suffix(".png")
    composite.save(out_path)
    return out_path


def main() -> None:
    """Scan input, process all images through all models, run post-passes."""
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / INPUT_FOLDER
    input_dir.mkdir(exist_ok=True)
    (base_dir / OUTPUT_BASE).mkdir(exist_ok=True)
    for m in MODELS:
        (base_dir / OUTPUT_BASE / m).mkdir(exist_ok=True)
    paths = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in INPUT_EXTENSIONS
    ]
    if not paths:
        print(f"No images found in {input_dir}. Add .png or .tiff files.")
        return
    print(f"Found {len(paths)} images. Loading models...")
    sessions: dict[str, Any] = {}
    for m in MODELS:
        if m == "sam2":
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                hf_id = SAM2_HF_IDS.get(SAM2_MODEL_SIZE, SAM2_HF_IDS["large"])
                print(f"  Loading SAM 2 ({SAM2_MODEL_SIZE})...")
                predictor = SAM2ImagePredictor.from_pretrained(hf_id)
                sessions["sam2_predictor"] = predictor
            except Exception as e:
                print(f"  Failed to load SAM 2: {e}")
                sessions["sam2_predictor"] = None
        else:
            from rembg import new_session

            sessions[m] = new_session(m)
    for model_name in MODELS:
        if model_name == "sam2" and sessions.get("sam2_predictor") is None:
            continue
        print(f"\nProcessing with {model_name}...")
        for i, img_path in enumerate(paths):
            print(f"  [{i+1}/{len(paths)}] {img_path.name}")
            process_image(img_path, model_name, sessions, base_dir)
    if ENABLE_COLOR_MATCHING:
        for model_name in MODELS:
            out_dir = base_dir / OUTPUT_BASE / model_name
            if not out_dir.exists():
                continue
            outputs = sorted(out_dir.glob("*"))
            outputs = [p for p in outputs if p.suffix.lower() in (".png", ".tiff", ".tif")]
            if len(outputs) < 2:
                continue
            ref = outputs[0]
            if isinstance(COLOR_MATCH_REFERENCE, str) and COLOR_MATCH_REFERENCE != "first":
                ref = base_dir / COLOR_MATCH_REFERENCE
                if not ref.exists():
                    ref = outputs[0]
            print(f"  Color matching {model_name} ({len(outputs)} images)...")
            color_match_batch(outputs, ref)
    if ENABLE_VAMPIRIC_CORRECTION:
        for model_name in MODELS:
            out_dir = base_dir / OUTPUT_BASE / model_name
            if not out_dir.exists():
                continue
            for out_path in out_dir.iterdir():
                if out_path.suffix.lower() not in (".png", ".tiff", ".tif"):
                    continue
                img = Image.open(out_path).convert("RGBA")
                if should_skip_vampiric(img, out_path.name):
                    continue
                result = apply_vampiric_correction(img)
                result.save(out_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
