"""
Character Cutout Workspace - V:tM NPC Image Processing Pipeline

Processes images through: upscale -> segment -> cutout -> alpha boost -> downscale ->
smart crop -> [colorize if needed] -> de-grade -> vampiric correction (skin-only, ref-driven).

Run from project root: python process_cutouts.py
"""

from __future__ import annotations

import os
import tempfile
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
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Compatibility: torchvision 0.17+ removed functional_tensor; basicsr still imports it
import sys
try:
    import torchvision.transforms.functional as _tv_f
    _ft = type(sys)("torchvision.transforms.functional_tensor")
    _ft.rgb_to_grayscale = _tv_f.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _ft
except ImportError:
    pass

import numpy as np
from PIL import Image
from tqdm import tqdm
from PIL.Image import Image as PILImage

# === CONFIGURATION (edit these) ===
# Environment: CUDA available. Model-size options default to largest/best for quality.
INPUT_FOLDER = "input"
OUTPUT_BASE = "output"
MODELS = ["birefnet-general", "birefnet-portrait", "bria-rmbg", "sam2"]
INPUT_EXTENSIONS = (".png", ".tiff", ".tif")

# Target height for all final cutouts (width scales proportionally)
TARGET_HEIGHT = 3000

# Draft mode: lower resolution for faster iteration (target 1000px, working 1500px)
DRAFT_MODE = True

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

# De-grading: neutralize color grading (sepia, blue cast) before color matching / vampiric
ENABLE_DE_GRADING = True
DE_GRADING_DARK_THRESHOLD = 60  # Mean luminance below this: apply exposure correction
DE_GRADING_EXPOSURE_GAMMA = 0.75  # < 1 brightens; applied only when image is very dark

# Optional colorization for near-grayscale images (Phase 0b)
ENABLE_COLORIZATION = False
COLORIZATION_SATURATION_THRESHOLD = 0.05

# Reference for vampiric skin look only (not used for clothing)
VAMPIRIC_REFERENCE = "color-references/male-vampire-5.png"

# Skin detection: "simple" (light-skin YCbCr fallback) | "precise" (AI human parsing)
SKIN_DETECTION_MODE = "precise"

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

# Effective target and working height (draft mode overrides to 1000 / 1500)
_EFFECTIVE_TARGET_HEIGHT = 1000 if DRAFT_MODE else TARGET_HEIGHT
WORKING_HEIGHT = int(_EFFECTIVE_TARGET_HEIGHT * UPSCALE_HEIGHT_FACTOR)

# Diagnostics: log which step each image is on (helps identify if stuck on upscale vs segment)
ENABLE_STEP_LOGGING = True

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


def _get_mean_saturation_rgb(rgb: np.ndarray, alpha: Optional[np.ndarray] = None) -> float:
    """Return mean saturation (0-1) over pixels. If alpha given, only foreground (alpha > 0.1)."""
    from skimage.color import rgb2hsv

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return 0.0
    hsv = rgb2hsv(rgb.astype(np.float64) / 255.0)
    sat = hsv[:, :, 1]
    if alpha is not None:
        mask = (alpha.astype(np.float64) / 255.0) > 0.1
        if not np.any(mask):
            return 0.0
        return float(np.mean(sat[mask]))
    return float(np.mean(sat))


_colorizer_instance: Any = None


def _get_colorizer() -> Any:
    """Lazy-load DeOldify colorizer; returns None if unavailable."""
    global _colorizer_instance
    if _colorizer_instance is not None:
        return _colorizer_instance
    try:
        from deoldify.visualize import get_image_colorizer  # pyright: ignore[reportMissingImports]

        (_MODEL_CACHE / "deoldify").mkdir(parents=True, exist_ok=True)
        _colorizer_instance = get_image_colorizer(
            root_folder=Path(_MODEL_CACHE) / "deoldify",
            render_factor=35,
            artistic=False,
        )
        return _colorizer_instance
    except ImportError:
        return None


def colorize_if_needed(img: PILImage) -> PILImage:
    """
    If image is nearly grayscale (mean saturation < threshold) and ENABLE_COLORIZATION,
    run DeOldify to restore plausible color. Otherwise return unchanged.
    Preserves alpha. Requires deoldify package (optional).
    """
    if not ENABLE_COLORIZATION:
        return img
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return img
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3:4] if arr.shape[2] == 4 else None
    mean_sat = _get_mean_saturation_rgb(rgb, alpha if alpha is not None else None)
    if mean_sat >= COLORIZATION_SATURATION_THRESHOLD:
        return img
    colorizer = _get_colorizer()
    if colorizer is None:
        return img
    fd, tmp_path_str = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    try:
        img_rgb = Image.fromarray(rgb)
        img_rgb.save(tmp_path)
        result = colorizer.get_transformed_image(
            tmp_path, render_factor=35, post_process=True, watermarked=False
        )
        result_arr = np.array(result.convert("RGB"))
        if alpha is not None:
            result_arr = np.dstack([result_arr, alpha])
        return Image.fromarray(result_arr)
    except Exception:
        return img
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def degrade_image(img: PILImage) -> PILImage:
    """
    Neutralize color grading (sepia, blue cast, etc.) via white balance and optional exposure correction.
    Affects entire image. Returns new image; does not modify original.
    """
    import cv2
    from skimage.exposure import adjust_gamma

    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return img
    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3:4].copy() if arr.shape[2] == 4 else None
    # OpenCV expects BGR
    bgr = rgb[:, :, ::-1]
    # Brighten very dark images before white balance (e.g. heavily blue-shifted dark photos)
    mean_lum = float(np.mean(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]))
    if mean_lum < DE_GRADING_DARK_THRESHOLD and DE_GRADING_EXPOSURE_GAMMA < 1.0:
        bgr_float = bgr.astype(np.float64) / 255.0
        bgr_float = adjust_gamma(bgr_float, DE_GRADING_EXPOSURE_GAMMA)
        bgr = (np.clip(bgr_float, 0, 1) * 255).astype(np.uint8)
    wb = cv2.xphoto.createGrayworldWB()
    result_bgr = wb.balanceWhite(bgr)
    result_rgb = result_bgr[:, :, ::-1]
    if alpha is not None:
        result = np.dstack([result_rgb, alpha])
    else:
        result = result_rgb
    return Image.fromarray(result)


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
    """Crop to content bounds (alpha >= threshold), add padding, re-scale to target height."""
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
    scale = _EFFECTIVE_TARGET_HEIGHT / pil_crop.height
    new_w = int(pil_crop.width * scale)
    return pil_crop.resize((new_w, _EFFECTIVE_TARGET_HEIGHT), Image.Resampling.LANCZOS)


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


def _get_ycbcr_skin_bounds() -> tuple[float, float, float, float, float, float]:
    """Return (y_min, y_max, cb_min, cb_max, cr_min, cr_max) for simple YCbCr fallback."""
    return (60.0, 255.0, 77.0, 127.0, 133.0, 173.0)


def _get_skin_coverage(img: PILImage) -> float:
    """Return fraction of foreground pixels that are detected as skin."""
    arr = np.array(img.convert("RGB"))
    if arr.ndim != 3:
        return 0.0
    if img.mode == "RGBA":
        alpha = np.array(img.split()[-1]) / 255.0
    else:
        alpha = np.ones((arr.shape[0], arr.shape[1]))
    fg = alpha > 0.1
    if not np.any(fg):
        return 0.0
    if SKIN_DETECTION_MODE == "precise":
        skin_mask = _get_skin_mask_precise(arr, img)
        skin = skin_mask > 32
    else:
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.169 * r - 0.331 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.419 * g - 0.081 * b
        ymn, ymx, cbmn, cbmx, crmn, crmx = _get_ycbcr_skin_bounds()
        skin = (
            (y > ymn) & (y < ymx) &
            (cb > cbmn) & (cb < cbmx) &
            (cr > crmn) & (cr < crmx)
        )
    skin_fg = np.sum(skin & fg)
    total_fg = np.sum(fg)
    if total_fg == 0:
        return 0.0
    return float(skin_fg) / float(total_fg)


def apply_vampiric_correction(
    img: PILImage,
    reference_path: Optional[str | Path] = None,
    base_dir: Optional[Path] = None,
) -> PILImage:
    """
    Apply skin-only adjustment toward reference vampiric look.
    If reference_path and base_dir given, LAB-match skin to reference skin stats.
    Otherwise use fixed HSV-based adjustment (paler, cooler).
    """
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return img
    rgb = arr[:, :, :3].astype(np.float64)
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else np.ones((arr.shape[0], arr.shape[1]), dtype=np.uint8) * 255
    skin_mask = _get_skin_mask(rgb, img)
    blend = skin_mask[:, :, np.newaxis].astype(np.float64) / 255.0
    from skimage.color import lab2rgb, rgb2lab

    if reference_path is not None and base_dir is not None:
        ref_mean, ref_std = _get_reference_skin_lab_stats(reference_path, base_dir)
        lab = rgb2lab(rgb / 255.0)
        skin_bool = skin_mask > 32
        if np.any(skin_bool):
            src_mean = np.array([np.mean(lab[skin_bool, i]) for i in range(3)])
            src_std = np.array([np.std(lab[skin_bool, i]) for i in range(3)])
            src_std = np.where(src_std < 1e-6, 1.0, src_std)
            for i in range(3):
                lab[skin_bool, i] = (lab[skin_bool, i] - src_mean[i]) / src_std[i] * ref_std[i] + ref_mean[i]
        rgb_adj = (np.clip(lab2rgb(lab), 0, 1) * 255).astype(np.uint8)
    else:
        from skimage.color import hsv2rgb, rgb2hsv

        hsv = rgb2hsv(rgb / 255.0)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        v_adj = v * VAMPIRIC_PALENESS
        h_adj = (h + VAMPIRIC_COOL_SHIFT) % 1.0
        s_adj = s * VAMPIRIC_SATURATION
        hsv_adj = np.stack([h_adj, s_adj, v_adj], axis=-1)
        rgb_adj = (hsv2rgb(hsv_adj) * 255).clip(0, 255).astype(np.uint8)
    result_rgb = (rgb * (1 - blend) + rgb_adj * blend).astype(np.uint8)
    if arr.shape[2] == 4:
        result = np.dstack([result_rgb, alpha])
    else:
        result = result_rgb
    return Image.fromarray(result)


# Fashn-human-parser skin labels: face, arms, hands, legs, feet, torso
_FASHN_SKIN_LABELS = frozenset({"face", "arms", "hands", "legs", "feet", "torso"})

_human_parser_pipeline: Any = None


def _get_human_parser() -> Any:
    """Lazy-load HuggingFace human parsing pipeline for full-body skin detection."""
    global _human_parser_pipeline
    if _human_parser_pipeline is not None:
        return _human_parser_pipeline
    try:
        from transformers import pipeline

        _human_parser_pipeline = pipeline(
            "image-segmentation",
            model="fashn-ai/fashn-human-parser",
        )
        return _human_parser_pipeline
    except Exception:
        return None


def _get_skin_mask_body(img: PILImage) -> np.ndarray:
    """
    Full-body skin mask using human parsing (fashn-human-parser). Returns 0-255 mask.
    Skin = face, arms, hands, legs, feet, torso.
    """
    pipe = _get_human_parser()
    if pipe is None:
        return np.zeros((img.height, img.width), dtype=np.uint8)
    try:
        results = pipe(img.convert("RGB"))
        h, w = np.array(img).shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for r in results:
            lbl = (r.get("label") or "").lower().strip()
            if lbl not in _FASHN_SKIN_LABELS:
                continue
            m = r.get("mask")
            if m is not None:
                m_arr = np.array(m)
                if m_arr.ndim == 3:
                    m_arr = m_arr[:, :, 0]
                if m_arr.shape[0] == h and m_arr.shape[1] == w:
                    mask = np.maximum(mask, m_arr)
                else:
                    from PIL import Image as _PIL
                    m_resized = np.array(_PIL.fromarray(m_arr).resize((w, h), _PIL.Resampling.LANCZOS))
                    if m_resized.ndim == 3:
                        m_resized = m_resized[:, :, 0]
                    mask = np.maximum(mask, m_resized)
        from skimage.filters import gaussian

        smooth = gaussian(mask.astype(np.float32), sigma=2)
        return (np.clip(smooth, 0, 255)).astype(np.uint8)
    except Exception:
        return np.zeros((img.height, img.width), dtype=np.uint8)


def _get_skin_mask_precise(rgb: np.ndarray, img: Optional[PILImage] = None) -> np.ndarray:
    """
    AI-based skin mask: union of human parsing (full-body). Returns 0-255.
    For full-body cutouts, human parser alone gives face + arms + hands + legs + feet + torso.
    """
    if img is None:
        img = Image.fromarray(rgb.astype(np.uint8))
    return _get_skin_mask_body(img)


def _get_skin_mask(rgb: np.ndarray, img: Optional[PILImage] = None) -> np.ndarray:
    """
    Skin mask 0-255. precise=AI (human parsing); simple/extended=YCbCr.
    """
    if SKIN_DETECTION_MODE == "precise":
        return _get_skin_mask_precise(rgb, img)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.169 * r - 0.331 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.419 * g - 0.081 * b
    ymn, ymx, cbmn, cbmx, crmn, crmx = _get_ycbcr_skin_bounds()
    skin = (
        (y > ymn) & (y < ymx) &
        (cb > cbmn) & (cb < cbmx) &
        (cr > crmn) & (cr < crmx)
    )
    from skimage.filters import gaussian

    smooth = gaussian(skin.astype(np.float32), sigma=2)
    return (np.clip(smooth, 0, 1) * 255).astype(np.uint8)


def _get_reference_skin_lab_stats(reference_path: str | Path, base_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load reference image and extract LAB mean/std from skin pixels.
    Returns (ref_mean, ref_std) for L, a, b channels.
    """
    from skimage.color import rgb2lab

    ref_full = base_dir / reference_path if isinstance(reference_path, str) else reference_path
    if not ref_full.is_file() or not ref_full.exists():
        return np.array([60.0, 10.0, 5.0]), np.array([20.0, 10.0, 10.0])  # Fallback
    ref_img = np.array(Image.open(ref_full).convert("RGB"))
    ref_rgb = ref_img.astype(np.float64) / 255.0
    ref_lab = rgb2lab(ref_rgb)
    skin_mask = _get_skin_mask(ref_img, Image.fromarray(ref_img))
    skin_bool = skin_mask > 32
    if not np.any(skin_bool):
        return np.array([np.mean(ref_lab[:, :, i]) for i in range(3)]), np.array(
            [np.std(ref_lab[:, :, i]) for i in range(3)]
        )
    ref_mean = np.array([np.mean(ref_lab[skin_bool, i]) for i in range(3)])
    ref_std = np.array([np.std(ref_lab[skin_bool, i]) for i in range(3)])
    ref_std = np.where(ref_std < 1e-6, 1.0, ref_std)
    return ref_mean, ref_std


def _log_step(msg: str) -> None:
    """Print step message; works alongside tqdm without breaking progress bar."""
    if ENABLE_STEP_LOGGING:
        tqdm.write(msg)


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
        tqdm.write(f"  Error loading {img_path}: {e}")
        return None
    orig_size = img.size
    working_h = WORKING_HEIGHT
    if img.height > working_h:
        scale = working_h / img.height
        img = img.resize((int(img.width * scale), working_h), Image.Resampling.LANCZOS)
    elif img.height < working_h:
        _log_step(f"    Upscaling {img_path.name} ({img.height} -> {working_h}px)...")
        img = upscale_image(img, working_h)
    seg_input = adjust_exposure(img, EXPOSURE_GAMMA) if ENABLE_EXPOSURE_ADJUST else img
    _log_step(f"    Segmenting {img_path.name} with {model_name}...")
    mask = generate_mask(seg_input, model_name, sessions)
    mask = mask.resize(img.size, Image.Resampling.LANCZOS)
    composite = Image.new("RGBA", img.size, (0, 0, 0, 0))
    composite.paste(img, (0, 0))
    composite.putalpha(mask)
    if ENABLE_ALPHA_BOOST:
        composite = alpha_boost(composite)
    scale = _EFFECTIVE_TARGET_HEIGHT / composite.height
    composite = composite.resize(
        (int(composite.width * scale), _EFFECTIVE_TARGET_HEIGHT),
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


def _print_gpu_status() -> None:
    """Print CUDA/GPU availability for Real-ESRGAN and rembg diagnostics."""
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            dev = torch.cuda.get_device_name(0)
            print(f"  CUDA: available ({dev})")
        else:
            print("  CUDA: not available - expect slow CPU processing")
    except ImportError:
        print("  CUDA: torch not loaded yet")
    try:
        import onnxruntime as ort

        provs = ort.get_available_providers()
        if "CUDAExecutionProvider" in provs:
            print("  ONNX Runtime: CUDA provider available")
        else:
            print("  ONNX Runtime: CUDA provider not available - rembg will use CPU")
    except ImportError:
        print("  ONNX Runtime: not checked (rembg uses its own)")


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
    if DRAFT_MODE:
        print(f"  DRAFT_MODE: target={_EFFECTIVE_TARGET_HEIGHT}px, working={WORKING_HEIGHT}px")
    _print_gpu_status()
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

            try:
                sessions[m] = new_session(m, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            except TypeError:
                sessions[m] = new_session(m)
    for model_name in MODELS:
        if model_name == "sam2" and sessions.get("sam2_predictor") is None:
            continue
        print(f"\nProcessing with {model_name}...")
        for img_path in tqdm(paths, desc=model_name, unit="img"):
            process_image(img_path, model_name, sessions, base_dir)
    if ENABLE_COLORIZATION or ENABLE_DE_GRADING:
        for model_name in MODELS:
            out_dir = base_dir / OUTPUT_BASE / model_name
            if not out_dir.exists():
                continue
            outputs = [p for p in out_dir.iterdir() if p.suffix.lower() in (".png", ".tiff", ".tif")]
            if outputs:
                label = "Colorize + De-grade" if (ENABLE_COLORIZATION and ENABLE_DE_GRADING) else ("Colorize" if ENABLE_COLORIZATION else "De-grading")
                print(f"\n{label} {model_name}...")
                for out_path in tqdm(outputs, desc=model_name, unit="img"):
                    try:
                        img = Image.open(out_path).convert("RGBA")
                        if ENABLE_COLORIZATION:
                            img = colorize_if_needed(img)
                        if ENABLE_DE_GRADING:
                            img = degrade_image(img)
                        img.save(out_path)
                    except Exception as e:
                        print(f"    Failed {out_path.name}: {e}")
    if ENABLE_VAMPIRIC_CORRECTION:
        for model_name in MODELS:
            out_dir = base_dir / OUTPUT_BASE / model_name
            if not out_dir.exists():
                continue
            outputs = [p for p in out_dir.iterdir() if p.suffix.lower() in (".png", ".tiff", ".tif")]
            if outputs:
                print(f"\nVampiric correction {model_name}...")
                for out_path in tqdm(outputs, desc=model_name, unit="img"):
                    img = Image.open(out_path).convert("RGBA")
                    if should_skip_vampiric(img, out_path.name):
                        continue
                    result = apply_vampiric_correction(
                        img,
                        reference_path=VAMPIRIC_REFERENCE,
                        base_dir=base_dir,
                    )
                    result.save(out_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
