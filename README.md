# histopathology_fm

Histopathology foundation models for extracting tile-level embeddings from H&E whole-slide images, starting with [GenBio-PathFM](https://huggingface.co/genbio-ai/genbio-pathfm). The target use case is 10x Xenium data, where the H&E scan of the same section is provided as a separate `*_he_image.ome.tif`.

> The input must be the **8-bit RGB H&E image**. Xenium's fluorescence images (`morphology.ome.tif`, `morphology_focus.ome.tif`, `morphology_mip.ome.tif`) are DAPI/immunofluorescence and are rejected by the script.

## Files

| File | What it does |
| --- | --- |
| `genbio_pathfm.py` | The `GenBioPathFM` class: loads the model and turns a slide into tile embeddings. |
| `run_genbio.py` | Command-line script that runs the class on one slide and saves the results. |

## Installation

```bash
pip install torch torchvision transformers==4.57.1 timm huggingface_hub tifffile zarr imagecodecs pillow numpy
```

GenBio-PathFM is not gated on Hugging Face, so no login is needed. The weights are downloaded on first run.

**GPU note:** if you see `The NVIDIA driver on your system is too old` and the model loads on CPU, your PyTorch build needs a newer CUDA than your driver supports. Install a PyTorch build matching your driver, for example for a CUDA 12.8 driver:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Usage

### Command line

```bash
# Quick test: 10 tissue tiles, with tile PNGs saved for inspection
python run_genbio.py path/to/he_image.ome.tif --output results/genbio --max-patches 10 --save-patches

# Whole slide
python run_genbio.py path/to/he_image.ome.tif --output results/genbio_full
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--output` | `results/genbio` | Output folder |
| `--target-mpp` | `0.5` | Resolution (µm/px) the model sees |
| `--patch-size` | `224` | Tile size in pixels given to the model |
| `--max-background` | `0.35` | Skip tiles with more than this fraction of background |
| `--max-patches` | all | Stop after this many tissue tiles |
| `--batch-size` | `32` | Tiles per forward pass (lower it if the GPU runs out of memory) |
| `--device` | GPU if available | `cuda` or `cpu` |
| `--save-patches` | off | Save each kept tile (raw crop and resized) as PNG |

The slide's source resolution is set at the top of `run_genbio.py` (`SOURCE_MPP_X`, `SOURCE_MPP_Y`, currently the Xenium human pancreas FFPE H&E at ~0.2738 µm/px). Change these for a different slide.

### Python

```python
from genbio_pathfm import GenBioPathFM

model = GenBioPathFM()  # loads the model once
features, coordinates = model.get_embedding(
    "path/to/he_image.ome.tif",
    source_mpp=(0.2737653984781153, 0.2737746952056361),  # or None to read from OME metadata
    target_mpp=0.5,
    patch_size=224,
    max_background=0.35,
    max_patches=None,
    batch_size=32,
    patch_folder=None,
)
# features:    (N, 4608) float32
# coordinates: (N, 2) int, top-left (x, y) of each tile in H&E pixels
```

## How it works

Preprocessing follows the GenBio-PathFM paper (bioRxiv 2026.03.17.712534):

1. Open the full-resolution level of the (pyramidal) OME-TIFF lazily, without loading the whole slide into memory.
2. Cut non-overlapping tiles covering `patch_size × target_mpp` µm (about 409 px at 0.274 µm/px for 224 px at 0.5 µm/px) and resize them to `patch_size`.
3. Keep tissue tiles only, using the Virchow2 HSV filter (H 90–180, S 8–255, V 103–255 on OpenCV's scale).
4. Normalize with the model's RGB mean/std. No stain normalization.
5. Embed each tile. The model embeds R, G and B separately and concatenates their CLS tokens: 3 × 1536 = **4608** values per tile.

## Outputs

| File | Contents |
| --- | --- |
| `features.npy` | `(N, 4608)` tile embeddings |
| `coordinates.csv` | `x,y` top-left corner of each tile in H&E image pixels; row *i* matches row *i* of `features.npy` |
| `patches/` | Tile PNGs (only with `--save-patches`) |

Notes:

- Each embedding is the tile-level (CLS) feature. There is no slide-level embedding; pool the tile features (for example `features.mean(axis=0)`) if you need one.
- Coordinates are in H&E pixel space, not Xenium micron space. To match tiles with Xenium cells, transform them with the H&E alignment matrix from the Xenium output.

## License

See [LICENSE](LICENSE).
