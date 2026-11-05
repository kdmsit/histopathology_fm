"""Extract GenBio-PathFM features from an H&E slide.

Install:
    pip install torch torchvision transformers==4.57.1 timm huggingface_hub tifffile zarr imagecodecs pillow numpy

Use from Python:
    from genbio_pathfm import GenBioPathFM
    model = GenBioPathFM()
    features, coordinates = model.get_embedding("slide_he_image.ome.tif", max_patches=10)

Or run this file directly (edit IMAGE_PATH at the bottom first):
    python genbio_pathfm.py

GenBio-PathFM is not gated on Hugging Face, so no login is needed.

Preprocessing follows the GenBio-PathFM paper (bioRxiv 2026.03.17.712534):
224x224 tiles, the Virchow2 HSV tissue filter, and no stain normalization.
The slide must be the separate 8-bit RGB H&E scan, not a Xenium fluorescence image.
"""
import re
from pathlib import Path

import numpy as np
import tifffile
import torch
import zarr
from PIL import Image
from torchvision import transforms
from transformers import AutoModel


class GenBioPathFM:
    """Wraps the GenBio-PathFM model and turns an H&E slide into patch embeddings."""

    MODEL_NAME = "genbio-ai/genbio-pathfm"
    EMBEDDING_DIM = 4608  # R, G and B embedded separately: 3 x 1536 CLS tokens.
    MEAN = (0.697, 0.575, 0.728)
    STD = (0.188, 0.240, 0.187)

    # Tissue filter from the paper (adopted from Virchow2): a pixel is tissue if
    # Hue in [90, 180], Saturation in [8, 255] and Value in [103, 255], with hue
    # on OpenCV's 0-180 scale.
    TISSUE_HUE = (90, 180)
    TISSUE_SATURATION = (8, 255)
    TISSUE_VALUE = (103, 255)

    def __init__(self, device=None):
        """Load the pretrained model once.

        device: "cuda", "cpu", etc. Defaults to the GPU when one is available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading GenBio-PathFM on {self.device}", flush=True)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model = self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def get_embedding(
        self,
        slide_path,
        source_mpp=None,
        target_mpp=0.5,
        patch_size=224,
        max_background=0.35,
        max_patches=None,
        batch_size=32,
        patch_folder=None,
    ):
        """Tile a slide, drop background tiles and embed the rest.

        slide_path: path to the 8-bit RGB H&E .tif / .ome.tif.
        source_mpp: slide resolution in micrometers per pixel, as one number or
            an (x, y) pair. If None, it is read from the OME-XML metadata.
        target_mpp: resolution the model sees. Each tile covers
            patch_size * target_mpp micrometers of tissue.
        patch_size: tile size in pixels given to the model.
        max_background: tiles with more than this fraction of background are skipped.
        max_patches: stop after this many tissue tiles; None processes the whole slide.
        batch_size: tiles sent to the model at once.
        patch_folder: if set, saves each kept tile (raw crop and resized) as PNG.

        Returns (features, coordinates):
            features: float32 array of shape (N, 4608).
            coordinates: int array of shape (N, 2) with the (x, y) top-left corner
                of each tile in source-image pixels (not Xenium coordinates).
        """
        slide_path = Path(slide_path)
        if patch_folder is not None:
            patch_folder = Path(patch_folder)
            patch_folder.mkdir(parents=True, exist_ok=True)

        features = []
        coordinates = []
        batch, batch_xy = [], []

        def flush():
            if not batch:
                return
            inputs = torch.stack(batch).to(self.device)
            with torch.inference_mode():
                output = self.model(inputs)
            vectors = output.float().cpu().numpy()
            if vectors.shape != (len(batch), self.EMBEDDING_DIM) or not np.isfinite(vectors).all():
                raise ValueError("Unexpected model output")
            features.extend(vectors)
            coordinates.extend(batch_xy)
            batch.clear()
            batch_xy.clear()
            print(f"Processed {len(features)} patches", flush=True)

        with tifffile.TiffFile(slide_path) as tif:
            mpp_x, mpp_y = self._resolve_mpp(tif, source_mpp)
            # About 409 source pixels become 224 pixels at 0.5 um/px for a 0.274 um/px slide.
            crop_width = round(patch_size * target_mpp / mpp_x)
            crop_height = round(patch_size * target_mpp / mpp_y)

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

                kept = 0
                done = False
                # Visit nonoverlapping tiles, skipping incomplete edges.
                for y in range(0, height - crop_height + 1, crop_height):
                    for x in range(0, width - crop_width + 1, crop_width):
                        pixels = np.asarray(slide[y:y + crop_height, x:x + crop_width, :])
                        patch = Image.fromarray(pixels).resize(
                            (patch_size, patch_size), Image.Resampling.BILINEAR
                        )
                        tissue_fraction = self._tissue_fraction(patch)
                        if 1 - tissue_fraction > max_background:
                            continue

                        tensor = self.transform(patch)
                        if patch_folder is not None:
                            name = f"patch_{kept:04d}_x{x}_y{y}"
                            Image.fromarray(pixels).save(patch_folder / f"{name}_raw.png")
                            patch.save(patch_folder / f"{name}_{patch_size}.png")
                            print(f"{name}: raw crop {pixels.shape}, resized {patch.size}, "
                                  f"tissue {tissue_fraction:.2f}, "
                                  f"min {tensor.min().item():.2f}, max {tensor.max().item():.2f}, "
                                  f"channel means {tensor.mean(dim=(1, 2)).tolist()}", flush=True)

                        batch.append(tensor)
                        batch_xy.append((x, y))
                        kept += 1
                        if len(batch) >= batch_size:
                            flush()
                        if max_patches is not None and kept >= max_patches:
                            done = True
                            break
                    if done:
                        break
                flush()

        if not features:
            raise ValueError("No tissue patches found; check the image and max_background")
        return np.stack(features).astype(np.float32), np.array(coordinates, dtype=np.int64)

    def _tissue_fraction(self, patch):
        """Fraction of pixels that pass the paper's HSV tissue filter."""
        # PIL stores hue as 0-255, so convert it to OpenCV's 0-180 scale.
        hsv = np.asarray(patch.convert("HSV")).astype(np.float32)
        hue = hsv[:, :, 0] * 180 / 255
        saturation, value = hsv[:, :, 1], hsv[:, :, 2]
        is_tissue = (
            (hue >= self.TISSUE_HUE[0]) & (hue <= self.TISSUE_HUE[1])
            & (saturation >= self.TISSUE_SATURATION[0]) & (saturation <= self.TISSUE_SATURATION[1])
            & (value >= self.TISSUE_VALUE[0]) & (value <= self.TISSUE_VALUE[1])
        )
        return float(is_tissue.mean())

    @staticmethod
    def _resolve_mpp(tif, source_mpp):
        """Return (mpp_x, mpp_y) from the argument or from the OME-XML metadata."""
        if source_mpp is not None:
            if np.isscalar(source_mpp):
                return float(source_mpp), float(source_mpp)
            return float(source_mpp[0]), float(source_mpp[1])
        ome = tif.ome_metadata or ""
        found_x = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', ome)
        found_y = re.search(r'PhysicalSizeY="([0-9.eE+-]+)"', ome)
        if not (found_x and found_y):
            raise ValueError("Could not read the slide resolution; pass source_mpp")
        # Assumes micrometers, the OME default unit.
        return float(found_x.group(1)), float(found_y.group(1))


if __name__ == "__main__":
    # Edit these to match the H&E file on your machine.
    IMAGE_PATH = Path("../../project/Xenium/Xenium_V1_human_Pancreas_FFPE_he_image.ome.tif")
    OUTPUT_FOLDER = Path("results/genbio")

    model = GenBioPathFM()
    features, coordinates = model.get_embedding(
        IMAGE_PATH,
        # Read from the pancreas H&E file's metadata; set to None to read it automatically.
        source_mpp=(0.2737653984781153, 0.2737746952056361),
        target_mpp=0.5,
        patch_size=224,
        max_background=0.35,
        max_patches=10,  # Start with 10; set to None to process the whole slide.
        patch_folder=OUTPUT_FOLDER / "patches",
    )

    # Row i in features.npy corresponds to row i in coordinates.csv.
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_FOLDER / "features.npy", features)
    np.savetxt(OUTPUT_FOLDER / "coordinates.csv", coordinates,
               delimiter=",", header="x,y", comments="", fmt="%d")
    print(f"Saved {len(features)} feature vectors to {OUTPUT_FOLDER}")
