"""Train the pool model on the checked frames of a labeling dataset (docs/pool-model.md).

Each run gets its own folder, `<out>/<date-time>/`:

    model.pt       the trained network (load with `PoolModel.load`)
    summary.json   settings, which frames trained and which validated, water IoU of each validation frame
    log.csv        loss per epoch, validation IoU every few epochs
    val/<name>.jpg each validation frame with the model's water (blue), wrong water (red) and missed water (yellow)
    events.out.tfevents.*  the same progress for TensorBoard, live while training:
                   .venv/bin/tensorboard --logdir data/models/pool   (then http://localhost:6006)
"""

import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from swim.labeling.dataset import Dataset, Frame
from swim.labeling.labels import FrameLabels
from swim.pool_model.data import CLASSES, IGNORE, TrainingSet, checked_frames, split, water_mask
from swim.pool_model.network import UNet
from swim.pool_model.predict import pick_device, predict_classes

EVAL_EVERY = 5  # epochs between validation checks (log, TensorBoard)


def train(dataset_root: str | Path, out_root: str | Path, *, epochs: int = 40, batch: int = 8,
          size: tuple[int, int] = (1024, 576), lr: float = 1e-3, encoder: str = "resnet18", pretrained: bool = True,
          val_every: int = 6, device: str | None = None, workers: int = 4, seed: int = 0, log=None) -> Path:
    """Train a water model; returns the run's folder. `lr` is the decoder's learning rate; the pretrained encoder
    learns 10× slower (it already knows images; with few frames it would forget that)."""
    if size[0] % 32 or size[1] % 32:
        raise ValueError(f"Input sides must be divisible by 32, got {size[0]}×{size[1]}")
    log = log or (lambda message: print(message, flush=True))  # flushed: progress shows in a log file too
    dataset = Dataset(dataset_root)
    frames = checked_frames(dataset)
    if len(frames) < 2:
        raise ValueError(f"{dataset.root}: {len(frames)} checked frames, need at least 2 (check frames on the Labels page)")
    labels_of = {frame.name: labels for frame, labels in frames}
    train_frames, val_frames = split([frame for frame, _ in frames], val_every)
    train_set = [(f, labels_of[f.name]) for f in train_frames]
    val_set = [(f, labels_of[f.name]) for f in val_frames]

    run = Path(out_root) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    torch.manual_seed(seed)
    dev = pick_device(device)
    log(f"{len(train_set)} training frames, {len(val_set)} validation frames; {encoder}, {size[0]}×{size[1]}, on {dev}")
    board = _tensorboard(run, log)

    net = UNet(len(CLASSES), encoder, pretrained).to(dev)
    batch = min(batch, len(train_set))
    loader = torch.utils.data.DataLoader(
        TrainingSet(dataset, train_set, size), batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
        persistent_workers=workers > 0, worker_init_fn=_quiet_worker)
    optimizer = torch.optim.AdamW([{"params": net.encoder_parameters(), "lr": lr / 10},
                                   {"params": net.decoder_parameters(), "lr": lr}], weight_decay=1e-4)
    steps = epochs * len(loader)
    warmup = min(50, max(1, steps // 10))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda k: (k + 1) / warmup if k < warmup else 0.5 * (1 + math.cos(math.pi * (k - warmup) / max(1, steps - warmup))))

    rows, start, step = [], time.time(), 0
    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        for images, masks in loader:
            images, masks = images.to(dev), masks.to(dev)
            loss = segmentation_loss(net(images), masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            losses.append(loss.item())
            step += 1
            if board:
                board.add_scalar("loss/train (step)", losses[-1], step)
                board.add_scalar("learning rate/decoder", optimizer.param_groups[1]["lr"], step)
        row = {"epoch": epoch, "loss": round(float(np.mean(losses)), 5), "val_iou": "", "seconds": round(time.time() - start)}
        if board:
            board.add_scalar("loss/train (epoch mean)", row["loss"], epoch)
        if val_set and (epoch % EVAL_EVERY == 0 or epoch == epochs):
            ious, pictures = evaluate(net, dataset, val_set, size, dev, draw=board is not None)
            row["val_iou"] = round(float(np.mean(list(ious.values()))), 4)
            if board:
                board.add_scalar("water IoU/validation mean", row["val_iou"], epoch)
                board.add_scalar("water IoU/validation worst", min(ious.values()), epoch)
                board.add_image("validation frames", _grid(pictures), epoch, dataformats="HWC")
        if board:
            board.flush()
        rows.append(row)
        log(f"epoch {epoch}/{epochs}  loss {row['loss']:.4f}" + (f"  val water IoU {row['val_iou']:.3f}" if row["val_iou"] != "" else "")
            + f"  {row['seconds']} s")

    torch.save({"network": {k: v.cpu() for k, v in net.state_dict().items()}, "classes": list(CLASSES),
                "encoder": encoder, "size": list(size)}, run / "model.pt")
    ious, pictures = evaluate(net, dataset, val_set, size, dev, draw=True)
    (run / "val").mkdir()
    for name, picture in pictures.items():
        cv2.imwrite(str(run / "val" / f"{name}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if board:
        board.close()
    with open(run / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"), "dataset": str(dataset.root), "classes": list(CLASSES),
        "encoder": encoder, "pretrained": pretrained, "size": list(size), "epochs": epochs, "batch": batch, "lr": lr,
        "seed": seed, "device": str(dev), "seconds": round(time.time() - start),
        "val_iou_mean": round(float(np.mean(list(ious.values()))), 4) if ious else None,
        "val_iou_min": round(min(ious.values()), 4) if ious else None,
        "val_iou": {name: round(iou, 4) for name, iou in sorted(ious.items(), key=lambda kv: kv[1])},
        "train_frames": [f.name for f in train_frames], "val_frames": [f.name for f in val_frames],
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    return run


def segmentation_loss(scores: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Cross-entropy plus Dice on the non-"other" classes. Cross-entropy learns every pixel; Dice weighs a class by
    its share of the frame, so small classes (lane ropes, later) count. IGNORE pixels count for neither."""
    ce = F.cross_entropy(scores, masks, ignore_index=IGNORE)
    valid = (masks != IGNORE).unsqueeze(1)
    probs = scores.softmax(1) * valid
    target = F.one_hot(masks.clamp(max=scores.shape[1] - 1), scores.shape[1]).permute(0, 3, 1, 2) * valid
    inter = (probs * target).sum((0, 2, 3))[1:]
    total = (probs + target).sum((0, 2, 3))[1:]
    dice = 1 - ((2 * inter + 1) / (total + 1)).mean()
    return ce + dice


def evaluate(net: UNet, dataset: Dataset, frames: list[tuple[Frame, FrameLabels]], size: tuple[int, int],
             device: torch.device, draw: bool = False) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Water IoU of each frame, at the frame's full size (the model's water against the labels' water), and with
    `draw` each frame with the differences drawn on it (`difference_picture`)."""
    ious, pictures = {}, {}
    for frame, labels in frames:
        image = cv2.imread(str(dataset.image_path(frame.name)))
        truth = water_mask(labels).astype(bool)
        water = predict_classes(net, image, size, device) == 1
        union = (truth | water).sum()
        ious[frame.name] = float((truth & water).sum() / union) if union else 1.0
        if draw:
            pictures[frame.name] = difference_picture(image, truth, water, frame.name, ious[frame.name])
    return ious, pictures


def difference_picture(image: np.ndarray, truth: np.ndarray, water: np.ndarray, name: str, iou: float) -> np.ndarray:
    """The frame with the model's water tinted blue, water where the labels have none red, missed water yellow."""
    view = image.astype(np.float32)
    for area, color in ((water & truth, (255, 110, 0)), (water & ~truth, (40, 40, 255)), (~water & truth, (0, 220, 255))):
        view[area] = view[area] * 0.55 + np.array(color, np.float32) * 0.45
    view = view.astype(np.uint8)
    scale = 960 / view.shape[1]
    view = cv2.resize(view, (960, round(view.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    text = f"{name}  water IoU {iou:.3f}   blue: water  red: wrongly water  yellow: missed"
    cv2.rectangle(view, (0, 0), (view.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(view, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def _tensorboard(run: Path, log):
    """A TensorBoard writer into the run's folder, or None if TensorBoard is not installed."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        log("TensorBoard not installed (.venv/bin/pip install -e \".[train]\"): progress only here and in log.csv")
        return None
    return SummaryWriter(log_dir=str(run), flush_secs=10)


def _grid(pictures: dict[str, np.ndarray], columns: int = 4, width: int = 320) -> np.ndarray:
    """The validation pictures side by side (RGB, for TensorBoard), in name order so each frame keeps its place."""
    tiles = [cv2.resize(p, (width, round(p.shape[0] * width / p.shape[1])), interpolation=cv2.INTER_AREA)
             for _, p in sorted(pictures.items())]
    height = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % columns)
    rows = [np.hstack(tiles[k:k + columns]) for k in range(0, len(tiles), columns)]
    return cv2.cvtColor(np.vstack(rows), cv2.COLOR_BGR2RGB)


def _quiet_worker(_: int) -> None:
    cv2.setNumThreads(1)  # several data loader workers: one OpenCV thread each
