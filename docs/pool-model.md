# The pool model (pool segmentation network)

_Agreed: 2026-11-16. Status: training built (water / other); used by the pipeline's pool stage (stage 2, since
2026-11-17)._

A small, fast network that finds the pool's **water** in every frame. It is trained on the frames checked on the Labels
page (`docs/labeling.md`). It replaces SAM3-LiteText (about 1.7 s per frame) with about 25 ms per frame. Lane ropes
come later as a third class. The model choice is explained in `research/pool-model-training.md`.

## Classes

| Number | Class | From the labels |
|---|---|---|
| 0 | other | everything outside the water polygons |
| 1 | water | inside a water polygon, or inside a lane rope (ropes lie in the pool) |

The masks are drawn from the label files with the same `rasterize()` as `python -m swim.labeling masks`, so training
always uses the labels as they are now, without a separate step. Only checked frames (`fixed`, `accepted`) are used.

## Network (`swim/pool_model/network.py`)

- **U-Net** on a **ResNet18** encoder pretrained on ImageNet (torchvision weights, downloaded once, 45 MB). ResNet34
  is available with `--encoder resnet34`.
- **Input**: the frame scaled so its long side is 1024, shape kept. That is 1024×576 for our landscape frames and
  576×1024 for upright ones (IMG_4854). Sides are rounded to multiples of 32, because the encoder halves the image 5
  times; the network takes any such size.
- **Training views** are 1024×576 at that same scale, zoomed 0.8–1.6×. Upright frames and quarter-turned views
  don't fill a landscape view, and the empty part is left out of the loss.
- **Output**: one score per class for every input pixel. For a frame, the scores are scaled back to the frame's size
  before picking the class.
- Why this size: the water would be found at 512×288 too, but ropes (later) are about 6 px wide at 1024×576 and 3 px
  at 512×288 (`research/pool-model-training.md`).

## Training (`swim/pool_model/train.py`, `data.py`)

- **Split**: every 6th frame of each video, in time order, validates; the rest train. The first set (91 frames) is 75
  training and 16 validation frames, and every video with 2 or more frames is in the validation set.
  - Frames of one video are 2 s or more apart, but they are alike, so validation is somewhat optimistic.
  - The real test is whole videos.
- **Augmentation**: a new random view of each training frame every epoch, aimed at what broke the earlier models:
  - mirror, zoom (0.8–1.6×), shift, tilt up to 10°;
  - a quarter turn in 15% of views (phone held sideways);
  - brightness, contrast, saturation and hue (sunlit water, glare);
  - motion blur (fast pans);
  - noise.

  Pixels from outside the frame (after zooming out or turning) are left out of the loss.
- **Loss**: cross-entropy plus Dice on the water class. Dice weighs a class by its share of the frame, which matters
  once ropes (a small share) are added.
- **Optimizer**: AdamW with learning rate 1e-3 for the decoder and 1e-4 for the pretrained encoder. With few frames,
  a fast-changing encoder would forget what it knows about images.
- **Schedule**: 50 warm-up steps, then a cosine decay to 0.
- **Defaults**: 40 epochs, batch 8, about 5 minutes on an M4 Pro (GPU, `mps`).
- **Validation**: water IoU of each validation frame at the frame's full size (the model's water against the labels'
  water), every 5 epochs and at the end.

## Files

`python -m swim.pool_model train` makes one folder per run, `data/models/pool/<date-time>/` (git-ignored):

```
model.pt        the trained network: weights, classes, encoder, input size
summary.json    settings, the training and validation frames, water IoU of each validation frame (worst first)
log.csv         loss per epoch, validation IoU every 10 epochs
val/<name>.jpg  each validation frame: the model's water blue, water where the labels have none red, missed water yellow
events.out.tfevents.*   the same progress for TensorBoard
```

**TensorBoard** shows the progress while training, and compares runs:
- the loss per step and per epoch;
- the learning rate;
- the validation water IoU (mean and worst frame) every 5 epochs;
- the validation frames with their differences drawn, every 5 epochs.

```bash
.venv/bin/tensorboard --logdir data/models/pool   # then http://localhost:6006
```

TensorBoard is the `train` extra (`.venv/bin/pip install -e ".[train]"`). Without it, training still runs, and the
progress shows only in the terminal and `log.csv`.

The run keeps the names of its training frames, so a model can be traced to its data.

## Commands

```bash
.venv/bin/python -m swim.pool_model train                       # defaults: 40 epochs, 1024x576, resnet18
.venv/bin/python -m swim.pool_model train --epochs 60 --encoder resnet34
.venv/bin/python -m swim.pool_model train --help                # all options
```

Using a model:

```python
from swim.pool_model.predict import PoolModel

model = PoolModel.load("data/models/pool/<run>/model.pt")
classes = model.predict(frame)  # BGR frame -> class number per pixel (0 other, 1 water), the frame's size
```

Like the other models, import it inside the function that creates the model; it imports torch.

## Results

First runs, 2026-11-16: 91 checked frames (75 training, 16 validation), defaults (40 epochs, ResNet18), 5 minutes on
an M4 Pro.

| Epoch | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|
| Validation water IoU, mean | 0.964 | 0.990 | 0.992 | 0.996 | **0.996** |

- **Worst frame 0.993.** The remaining differences are thin strips along the pool's edge, where the label's
  simplified outline and the model disagree by a few pixels.
- Correct on the phone held sideways (IMG_4887 f947: 0.993) and on the upright video (IMG_4854: 0.993).
- The run before the upright fix squeezed IMG_4854's frames into 1024×576: 0.982 on that frame, with a red sign
  behind the fence taken for water.
- Not tested yet: whole videos (motion blur, camera pointed at the floor), where the earlier models failed. The
  validation frames come from the same videos as the training frames.

## Next

- Run it on whole videos and look at where it fails. Add frames like those (pick, SAM3, check), train again.
- The pipeline's `pool` stage (`swim/stages/pool/`, `docs/pipeline.md`) runs the newest model on every frame
  (`pool.model`: a run folder name, or `latest`). Next: filtering drops boxes outside the water.
- Lane ropes as a third class, for the lanes (joining broken tracks).
