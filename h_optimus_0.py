"""Extract H-Optimus-0 features from your H&E slide.

Install:
    pip install torch torchvision transformers==4.57.1 timm huggingface_hub tifffile zarr imagecodecs pillow numpy
Run:
    python h_optimus_0.py

Edit IMAGE_PATH below to match the H&E file on your server.
H-Optimus-0 users: accept access on Hugging Face, then run `hf auth login`.
"""
from pathlib import Path

import numpy as np
import tifffile
import torch
import zarr
from PIL import Image
from torchvision import transforms
import timm


# STEP 1: Set the input image, output folder, and patch settings.
# This must point to the separate H&E scan, not a Xenium fluorescence image.
IMAGE_PATH = Path("../project/Xenium/Xenium_V1_human_Pancreas_FFPE_he_image.ome.tif")
OUTPUT_FOLDER = Path("results/hoptimus")

# These source resolutions were read from YOUR pancreas H&E file's metadata.
# Change them if you use a different slide. Units: micrometers per pixel.
SOURCE_MPP_X = 0.2737653984781153
SOURCE_MPP_Y = 0.2737746952056361
TARGET_MPP = 0.5
PATCH_SIZE = 224
MIN_TISSUE = 0.20  # Keep patches with at least 20% colored tissue.
MAX_PATCHES = 10   # Start with 10; set to None to process the whole slide.


def main():
    # STEP 2: Select the GPU if available and load the pretrained model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading H-Optimus-0 on {device}", flush=True)
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    model = model.to(device).eval()

    # STEP 3: Apply this model's required RGB normalization.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
    ])

    # STEP 4: Calculate how much original tissue belongs in each patch.
    # About 409 source pixels become 224 pixels at 0.5 micrometers/pixel.
    crop_width = round(PATCH_SIZE * TARGET_MPP / SOURCE_MPP_X)
    crop_height = round(PATCH_SIZE * TARGET_MPP / SOURCE_MPP_Y)
    features = []
    coordinates = []

    # STEP 5: Open the full-resolution TIFF with tiled reading to save RAM.
    with tifffile.TiffFile(IMAGE_PATH) as tif:
        series = tif.series[0].levels[0]
        if series.axes not in ("YXS", "YXC") or series.shape[-1] != 3 or series.dtype != np.dtype("uint8"):
            raise ValueError("Expected an 8-bit RGB H&E TIFF")
        with series.aszarr() as store:
            slide = zarr.open(store, mode="r")
            # A pyramidal TIFF may open as a group of resolution levels.
            # Select its full-resolution image array before reading pixels.
            if isinstance(slide, zarr.Group):
                multiscales = slide.attrs.get("multiscales")
                level_path = multiscales[0]["datasets"][0]["path"] if multiscales else "0"
                slide = slide[level_path]
            height, width = slide.shape[:2]

            # STEP 6: Visit nonoverlapping patches, skipping incomplete edges.
            for y in range(0, height - crop_height + 1, crop_height):
                for x in range(0, width - crop_width + 1, crop_width):
                    pixels = np.asarray(slide[y:y+crop_height, x:x+crop_width, :])
                    patch = Image.fromarray(pixels).resize(
                        (PATCH_SIZE, PATCH_SIZE), Image.Resampling.BILINEAR
                    )

                    # STEP 7: Skip mostly white background using color saturation.
                    hsv = np.asarray(patch.convert("HSV"))
                    tissue_fraction = (hsv[:, :, 1] > 25).mean()
                    if tissue_fraction < MIN_TISSUE:
                        continue

                    # STEP 8: Extract one feature vector from this tissue patch.
                    inputs = transform(patch).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        output = model(inputs)
                    vector = output.float().cpu().numpy()[0]
                    if vector.shape != (1536,) or not np.isfinite(vector).all():
                        raise ValueError("Unexpected model output")
                    features.append(vector)
                    coordinates.append((x, y))
                    print(f"Processed {len(features)} patches", flush=True)
                    if MAX_PATCHES is not None and len(features) >= MAX_PATCHES:
                        break
                if MAX_PATCHES is not None and len(features) >= MAX_PATCHES:
                    break

    # STEP 9: Save features and their matching original-image coordinates.
    # Row i in features.npy corresponds to row i in coordinates.csv.
    # Coordinates use source-image pixels, not Xenium cell coordinates.
    if not features:
        raise ValueError("No tissue patches found; check the image and MIN_TISSUE")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_FOLDER / "features.npy", np.stack(features))
    np.savetxt(OUTPUT_FOLDER / "coordinates.csv", coordinates,
               delimiter=",", header="x,y", comments="", fmt="%d")
    print(f"Saved {len(features)} feature vectors to {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
