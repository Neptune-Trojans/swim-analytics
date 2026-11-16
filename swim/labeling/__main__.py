"""Make training data for the pool segmentation model (docs/labeling.md). Run from the repo root:

    python -m swim.labeling pick data/input/*.MOV        # 1. pick frames (one every 2 s, at most 25 per video)
    python -m swim.labeling auto                          # 2. first labels from SAM3-LiteText (about 2 s per frame)
    python -m swim.app                                    # 3. check and fix them on the Labels page
    python -m swim.labeling masks                         # 4. training masks from the checked labels
    python -m swim.labeling status                        # how many frames are checked
"""

import argparse
from pathlib import Path

from swim.labeling.commands import auto_label, pick, write_masks
from swim.labeling.dataset import Dataset

DEFAULT_DATASET = "data/labels/pool"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m swim.labeling", description="Make training data for the pool model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help=f"Dataset folder (default: {DEFAULT_DATASET})")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("pick", help="Pick frames from videos")
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--every", type=float, default=2.0, help="Seconds between picked frames (default 2)")
    p.add_argument("--max-per-video", type=int, default=25, help="At most this many frames per video (default 25)")

    a = commands.add_parser("auto", help="First labels from SAM3-LiteText for frames that have none")
    a.add_argument("--redo", action="store_true", help="Label every frame again (checked labels are kept)")

    m = commands.add_parser("masks", help="Training masks (PNG) from the checked labels")
    m.add_argument("--all", action="store_true", help="Also frames not checked yet")

    commands.add_parser("status", help="How many frames are checked")
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    if args.command == "pick":
        pick(dataset, args.videos, args.every, args.max_per_video)
    elif args.command == "auto":
        auto_label(dataset, redo=args.redo)
    elif args.command == "masks":
        write_masks(dataset, include_unreviewed=args.all)
    else:
        counts = dataset.status_counts()
        print(f"{dataset.root}: {sum(counts.values())} frames: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
