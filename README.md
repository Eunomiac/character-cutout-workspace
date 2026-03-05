# Character Cutout Workspace

A command-line Python workflow for producing **Vampire: the Masquerade** NPC character cutouts: background removal, vampiric skin correction (paler, cooler tones), and batch color matching for a consistent look across all cutouts.

## Features

- **Multi-model background removal** — BiRefNet-General, BiRefNet-Portrait, BRIA RMBG 2.0, and SAM 2
- **Target-height output** — Uniform cutout size (default 3000px height)
- **Upscale → cutout → downscale pipeline** — Better mask quality by giving models more resolution to work with
- **Exposure pre-processing** — Gamma brightening for segmentation only (final cutout unchanged)
- **Alpha matting** — Refined edges for hair and fine detail
- **Alpha boost** — Fixes semi-transparent “see-through” patches
- **Smart crop** — Trims ghost particles at edges, re-scales to target height
- **Batch color matching** — LAB-based normalization so outputs share a consistent look
- **Vampiric skin correction** — Skin-only paler/cooler adjustment; skipped for inhuman/masked characters

## Requirements

- **Python** 3.8+
- **CUDA** GPU (recommended)
- **Windows** 64-bit (tested on Windows 11)

## Installation

1. **Install PyTorch with CUDA** (if using GPU):

   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

2. **Create and activate a virtual environment**:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:

   ```powershell
   pip install -r requirements.txt
   ```

   On first run, rembg, SAM 2, and Real-ESRGAN will download their model weights (~500MB+ total).

## Project Structure

```text
├── input/                  # Drop .png, .tiff, .tif source images here
├── output/                 # Cutouts saved by model:
│   ├── birefnet-general/
│   ├── birefnet-portrait/
│   ├── bria-rmbg/
│   └── sam2/
├── color-references/       # Optional: reference image(s) for color matching
├── process_cutouts.py      # Main script
├── requirements.txt
└── weights/                # Auto-created: Real-ESRGAN model weights
```

## Usage

1. Place source images in `input/` (`.png`, `.tiff`, or `.tif`).
2. Run the script from the project root:

   ```powershell
   python process_cutouts.py
   ```

3. Find cutouts in `output/<model_name>/` for each image.

## Configuration

Edit the config block at the top of `process_cutouts.py`:

### Paths & Models

| Parameter | Default | Description |
| ----------- | ----------- | ----------- |
| `INPUT_FOLDER` | `"input"` | Folder containing source images |
| `OUTPUT_BASE` | `"output"` | Base folder for cutout output |
| `MODELS` | `["birefnet-general", "birefnet-portrait", "bria-rmbg", "sam2"]` | Background-removal models to run |
| `INPUT_EXTENSIONS` | `(".png", ".tiff", ".tif")` | Accepted input file extensions |

### Size & Scaling

| Parameter | Default | Description |
| ----------- | ----------- | ----------- |
| `TARGET_HEIGHT` | `3000` | Output height in pixels (width scales proportionally) |
| `UPSCALE_HEIGHT_FACTOR` | `1.5` | Upscale to 1.5× target height before cutout for better masks |

### Exposure (Segmentation Only)

| Parameter | Default | Description |
| ----------- | ----------- | ----------- |
| `ENABLE_EXPOSURE_ADJUST` | `True` | Brighten a copy for segmentation; final cutout unchanged |
| `EXPOSURE_GAMMA` | `0.55` | Gamma < 1 brightens (0.5–0.6 for dark images) |

### SAM 2

| Parameter | Default | Description |
| ----------- | ----------- | ----------- |
| `SAM2_MODEL_SIZE` | `"large"` | Model size: `tiny`, `small`, `base_plus`, `large` |

### Alpha Matting (rembg models)

| Parameter | Default | Description |
| ----------- | ----------- | ----------- |
| `ENABLE_ALPHA_MATTING` | `True` | Refine edges for hair/wispy detail |
| `ALPHA_MATTING_FOREGROUND_THRESHOLD` | `240` | Foreground threshold |
| `ALPHA_MATTING_BACKGROUND_THRESHOLD` | `10` | Background threshold |
| `ALPHA_MATTING_ERODE_STRUCTURE_SIZE` | `10` | Erosion structure size |

### Alpha Boost

| Parameter | Default | Description |
| ----------- | ----------- | ----------- |
| `ENABLE_ALPHA_BOOST` | `True` | Fix semi-transparent patches (like “duplicate layer 4×, merge”) |
| `ALPHA_BOOST_PASSES` | `4` | Number of equivalent composite passes |

### Smart Crop

| Parameter | Default | Description |
| ----------- | ----------- | ----------- |
| `ENABLE_SMART_CROP` | `True` | Trim to content bounds, ignore low-alpha ghost particles |
| `CROP_ALPHA_THRESHOLD` | `0.4` | Pixels with alpha ≥ this count as content |
| `CROP_PADDING` | `5` | Extra pixels around crop |

### Batch Color Matching

| Parameter | Default | Description |
| ----------- | ----------- | ----------- |
| `ENABLE_COLOR_MATCHING` | `True` | Normalize brightness/color across outputs |
| `COLOR_MATCH_REFERENCE` | `"color-references/male-vampire-5.png"` | Path to reference image, or `"first"` to use first processed image |

### Vampiric Skin Correction

| Parameter | Default | Description |
| ----------- | ----------- | ----------- |
| `ENABLE_VAMPIRIC_CORRECTION` | `True` | Apply paler, cooler skin tone (skin only) |
| `VAMPIRIC_PALENESS` | `0.85` | 1.0 = no change; &lt; 1 = paler |
| `VAMPIRIC_COOL_SHIFT` | `0.08` | Hue shift toward blue (0 = none) |
| `VAMPIRIC_SATURATION` | `0.9` | Slight desaturation on skin (1.0 = no change) |

### Inhuman / Monster / Masked

| Parameter | Default | Description |
| ----------- | ----------- | ----------- |
| `INHUMAN_DETECTION` | `"auto"` | `"auto"` \| `"suffix"` \| `"none"` — when to skip vampiric correction |
| `INHUMAN_SKIN_THRESHOLD` | `0.08` | (auto) Skip if skin &lt; 8% of foreground |
| `INHUMAN_FILENAME_SUFFIX` | `"-inhuman"` | (suffix) Skip if filename contains this, e.g. `npc_orc-inhuman.png` |

## Color Reference Tips

- **Choose a neutral reference** — Avoid strong makeup (e.g. dark lipstick) so it doesn’t affect all characters.
- **Format** — PNG or high-quality JPG; LAB matching uses color statistics, not texture or style.

## License

See individual dependencies (rembg, SAM 2, Real-ESRGAN, etc.) for their licenses.
