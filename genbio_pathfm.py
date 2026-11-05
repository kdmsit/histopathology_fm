"""Extract GenBio-PathFM features from your H&E slide.

Install:
    pip install torch torchvision transformers==4.57.1 timm huggingface_hub tifffile zarr imagecodecs pillow numpy
Run:
    python genbio_pathfm.py

Edit IMAGE_PATH below to match the H&E file on your server.
GenBio-PathFM is not gated on Hugging Face, so no login is needed.

Preprocessing follows the GenBio-PathFM paper (bioRxiv 2026.03.17.712534):
224x224 tiles, the Virchow2 HSV tissue filter, and no stain normalization.
"""
from pathlib import Path

import numpy as np
import tifffile
import torch
import zarr
from PIL import Image
from torchvision import transforms
from transformers import AutoModel


# STEP 1: Set the input image, output folder, and patch settings.
# This must point to the separate H&E scan, not a Xenium fluorescence image.
IMAGE_PATH = Path("../../project/Xenium/Xenium_V1_human_Pancreas_FFPE_he_image.ome.tif")
OUTPUT_FOLDER = Path("results/genbio")

# These source resolutions were read from YOUR pancreas H&E file's metadata.
# Change them if you use a different slide. Units: micrometers per pixel.
SOURCE_MPP_X = 0.2737653984781153
SOURCE_MPP_Y = 0.2737746952056361

# Change it to focus on different aspects of the Cell
TARGET_MPP = 0.1
PATCH_SIZE = 224

# Tissue filter from the paper (adopted from Virchow2): a pixel is tissue if
# Hue in [90, 180], Saturation in [8, 255] and Value in [103, 255], with hue on
# OpenCV's 0-180 scale. Patches with more than 35% background are discarded.
TISSUE_HUE = (90, 180)
TISSUE_SATURATION = (8, 255)
TISSUE_VALUE = (103, 255)
MAX_BACKGROUND = 0.35
MAX_PATCHES = 10   # Start with 10; set to None to process the whole slide.
SAVE_PATCHES = True  # Save each kept patch as PNG images so you can inspect them.

# PATCH_FOLDER = OUTPUT_FOLDER / "patches_"+str(TARGET_MPP)
PATCH_FOLDER = OUTPUT_FOLDER / f"patches_{TARGET_MPP}"


def main():
    # Load the pretrained Gen-BioFM model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GenBio-PathFM on {device}", flush=True)
    model = AutoModel.from_pretrained( "genbio-ai/genbio-pathfm", trust_remote_code=True)
    model = model.to(device).eval()

    # STEP 3: Apply this model's required RGB normalization.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.697, 0.575, 0.728), std=(0.188, 0.240, 0.187)),
    ])

    # Calculate how much original tissue belongs in each patch.
    crop_width = round(PATCH_SIZE * TARGET_MPP / SOURCE_MPP_X)
    crop_height = round(PATCH_SIZE * TARGET_MPP / SOURCE_MPP_Y)
    features = []
    coordinates = []

    # Open the full-resolution TIFF with tiled reading to save RAM.
    with tifffile.TiffFile(IMAGE_PATH) as tif:
        series = tif.series[0].levels[0]
        if series.axes not in ("YXS", "YXC") or series.shape[-1] != 3 or series.dtype != np.dtype("uint8"):
            raise ValueError("Expected an 8-bit RGB H&E TIFF")
        with series.aszarr() as store:
            slide = zarr.open(store, mode="r")
            if isinstance(slide, zarr.Group):
                multiscales = slide.attrs.get("multiscales")
                level_path = multiscales[0]["datasets"][0]["path"] if multiscales else "0"
                slide = slide[level_path]
            height, width = slide.shape[:2]

            # Visit nonoverlapping patches, skipping incomplete edges.
            for y in range(0, height - crop_height + 1, crop_height):
                for x in range(0, width - crop_width + 1, crop_width):
                    pixels = np.asarray(slide[y:y+crop_height, x:x+crop_width, :])
                    patch = Image.fromarray(pixels).resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.BILINEAR)

                    # Skip background with the paper's HSV tissue filter.
                    # PIL stores hue as 0-255, so convert it to OpenCV's 0-180 scale.
                    hsv = np.asarray(patch.convert("HSV")).astype(np.float32)
                    hue = hsv[:, :, 0] * 180 / 255
                    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
                    is_tissue = (
                        (hue >= TISSUE_HUE[0]) & (hue <= TISSUE_HUE[1])
                        & (saturation >= TISSUE_SATURATION[0]) & (saturation <= TISSUE_SATURATION[1])
                        & (value >= TISSUE_VALUE[0]) & (value <= TISSUE_VALUE[1])
                    )
                    tissue_fraction = is_tissue.mean()
                    if 1 - tissue_fraction > MAX_BACKGROUND:
                        continue

                    # STEP 8: Save the patch before and after resizing, and print
                    # what the model receives after normalization.
                    inputs = transform(patch).unsqueeze(0).to(device)
                    if SAVE_PATCHES:
                        PATCH_FOLDER.mkdir(parents=True, exist_ok=True)
                        name = f"patch_{len(features):04d}_x{x}_y{y}"
                        Image.fromarray(pixels).save(PATCH_FOLDER / f"{name}_raw.png")
                        # patch.save(PATCH_FOLDER / f"{name}_224.png")
                        print(f"{name}: raw crop {pixels.shape}, resized {patch.size}, "
                              f"tissue {tissue_fraction:.2f}, model input {tuple(inputs.shape)}, "
                              f"min {inputs.min().item():.2f}, max {inputs.max().item():.2f}, "
                              f"channel means {inputs.mean(dim=(0, 2, 3)).tolist()}", flush=True)

                    # STEP 9: Extract one feature vector from this tissue patch.
                    # The model embeds R, G and B separately and concatenates their
                    # CLS tokens: 3 x 1536 = 4608 numbers per patch.
                    with torch.inference_mode():
                        output = model(inputs)
                    vector = output.float().cpu().numpy()[0]
                    if vector.shape != (4608,) or not np.isfinite(vector).all():
                        raise ValueError("Unexpected model output")
                    features.append(vector)
                    coordinates.append((x, y))
                    print(f"Processed {len(features)} patches", flush=True)
                    if MAX_PATCHES is not None and len(features) >= MAX_PATCHES:
                        break
                if MAX_PATCHES is not None and len(features) >= MAX_PATCHES:
                    break

    # STEP 10: Save features and their matching original-image coordinates.
    # Row i in features.npy corresponds to row i in coordinates.csv.
    # Coordinates use source-image pixels, not Xenium cell coordinates.
    if not features:
        raise ValueError("No tissue patches found; check the image and MAX_BACKGROUND")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_FOLDER / "features.npy", np.stack(features))
    np.savetxt(OUTPUT_FOLDER / "coordinates.csv", coordinates,
               delimiter=",", header="x,y", comments="", fmt="%d")
    print(f"Saved {len(features)} feature vectors to {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
