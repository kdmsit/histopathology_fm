"""Run GenBio-PathFM on an H&E slide and save the patch embeddings.

Example:
    python run_genbio.py ../../project/Xenium/Xenium_V1_human_Pancreas_FFPE_he_image.ome.tif \
        --output results/genbio --max-patches 10 --save-patches
"""
import argparse
from pathlib import Path

import numpy as np

from genbio_pathfm import GenBioPathFM

# Source resolution of the pancreas H&E slide, in micrometers per pixel.
# Change these if you use a different slide.
SOURCE_MPP_X = 0.2737653984781153
SOURCE_MPP_Y = 0.2737746952056361


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("slide", type=Path, help="8-bit RGB H&E .tif / .ome.tif")
    parser.add_argument("--output", type=Path, default=Path("results/genbio"), help="output folder")
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--max-background", type=float, default=0.35)
    parser.add_argument("--max-patches", type=int, default=None, help="omit to process the whole slide")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda or cpu; defaults to GPU if available")
    parser.add_argument("--save-patches", action="store_true", help="save each kept tile as PNG")
    args = parser.parse_args()

    model = GenBioPathFM(device=args.device)
    features, coordinates = model.get_embedding(
        args.slide,
        source_mpp=(SOURCE_MPP_X, SOURCE_MPP_Y),
        target_mpp=args.target_mpp,
        patch_size=args.patch_size,
        max_background=args.max_background,
        max_patches=args.max_patches,
        batch_size=args.batch_size,
        patch_folder=args.output / "patches" if args.save_patches else None,
    )

    # Row i in features.npy corresponds to row i in coordinates.csv.
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "features.npy", features)
    np.savetxt(args.output / "coordinates.csv", coordinates,
               delimiter=",", header="x,y", comments="", fmt="%d")
    print(f"Saved {len(features)} feature vectors of size {features.shape[1]} to {args.output}")


if __name__ == "__main__":
    main()
