"""Train the pool segmentation network (docs/pool-model.md). Run from the repo root:

    python -m swim.pool_model train      # on the checked frames of data/labels/pool; a run folder in data/models/pool/
"""

import argparse


def size(text: str) -> tuple[int, int]:
    width, _, height = text.lower().partition("x")
    try:
        return int(width), int(height)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, e.g. 1024x576, got '{text}'") from None


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m swim.pool_model", description="Train the pool segmentation network.")
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train", help="Train on the checked frames (water / other)")
    t.add_argument("--dataset", default="data/labels/pool", help="Labeling dataset (default: data/labels/pool)")
    t.add_argument("--out", default="data/models/pool", help="Folder for run folders (default: data/models/pool)")
    t.add_argument("--epochs", type=int, default=40, help="Passes over the training frames (default 40)")
    t.add_argument("--batch", type=int, default=8, help="Frames per training step (default 8)")
    t.add_argument("--size", type=size, default=(1024, 576), help="Network input, sides divisible by 32 (default 1024x576)")
    t.add_argument("--lr", type=float, default=1e-3, help="Learning rate of the decoder; the encoder's is 10× lower (default 1e-3)")
    t.add_argument("--encoder", default="resnet18", choices=["resnet18", "resnet34"], help="Pretrained encoder (default resnet18)")
    t.add_argument("--val-every", type=int, default=6, help="Every n-th frame of each video validates (default 6)")
    t.add_argument("--device", default="auto", help="mps, cuda, cpu or auto (default: the fastest here)")
    t.add_argument("--workers", type=int, default=4, help="Processes preparing training frames (default 4)")
    t.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from swim.pool_model.train import train  # imports torch

    try:
        run = train(args.dataset, args.out, epochs=args.epochs, batch=args.batch, size=args.size, lr=args.lr,
                    encoder=args.encoder, val_every=args.val_every, device=args.device, workers=args.workers, seed=args.seed)
    except ValueError as e:
        parser.exit(1, f"{e}\n")
    import json

    summary = json.loads((run / "summary.json").read_text())
    print(f"\nSaved {run}/model.pt ({summary['seconds']} s)")
    if summary["val_iou"]:
        print(f"Validation water IoU: mean {summary['val_iou_mean']:.3f}, worst {summary['val_iou_min']:.3f}")
        print("Worst frames: " + ", ".join(f"{name} {iou:.3f}" for name, iou in list(summary["val_iou"].items())[:3]))
        print(f"Pictures: {run}/val/")


if __name__ == "__main__":
    main()
