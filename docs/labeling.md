# Labeling (training data for the pool segmentation model)

_Agreed: 2026-11-16. Status: built (frame picking, SAM3 labels, Labels page, training masks); the model is not trained
yet._

Goal: a small, fast model that finds the pool's **water** and the **lane ropes** in every frame (lanes are the clue
for joining a swimmer's broken tracks, see `research/tracklet-linking.md`). SAM3-LiteText does this well but takes
about 2 s per frame on a MacBook (`research/pool-water-segmentation.md`), so it labels a set of frames once, we check
and fix its labels by hand, and a dedicated model is trained on them. All videos are filmed in the same pool, so a
model learned from them fits.

## Classes

| Number | Class | What it covers |
|---|---|---|
| 0 | other | everything outside the pool (deck, gutter, buildings, trees, sky, the sign, people on the deck) and any pixel that can't be decided |
| 1 | water | the pool and everything in it: water, splashes, foam, glare, reflections, the floor lines, and swimmers (cap, goggles and fins included) |
| 2 | lane rope | the floats of every rope, all colors |

When classes meet: lane rope > water > other. Not classes, on purpose: swimmers (they come from the detection
stage), floor lane lines, each rope on its own (separated afterwards by their straight lines), red vs blue rope
sections (from the rope colors along each rope).

## Labels are shapes, not pixels

- **Water**: polygons (the pool area, swimmers included). A sign or a person can split it into several. Polygons
  have no holes: everything inside the pool's outline counts as pool (a kickboard in the middle of the pool too).
- **Lane rope**: a polygon around each rope (its outline), usually 4 corners: a straight rope stays straight in the
  image, even in perspective, and its near end looks thicker than its far end. A swimmer crossing a rope doesn't cut
  it: one outline across. (Until 2026-11-16 a rope was a line with a width at each end; drawing ropes by clicking dots
  around them turned out simpler.)

Fixing shapes means dragging a few corners instead of painting pixels. Measured on 42 frames (2026-11-16): the pool area
as a polygon with about 50 corners (a corner wherever SAM3's outline bends more than 8 px) matches SAM3's mask within
IoU 0.99; the ropes as straight bands sit on SAM3's ropes (IoU 0.84, lower only because a rope is a few pixels wide).
With the clean-up below (floor lines added, gaps closed, notches at the frame's edge filled, corners along the
frame's edge dropped), the 91 frames of the first set have a median of 9 water corners (90% have 22 or fewer).

## Files

One folder per dataset, `data/labels/pool/` (git-ignored):

```
frames.json             the picked frames: name, video, frame index, size
images/<name>.jpg       the frames (JPEG, quality 95)
sam3/<name>.json        SAM3's labels, never edited (to measure how much was corrected)
labels/<name>.json      the labels you check and fix (start as a copy of sam3/)
masks/<name>.png        training masks drawn from labels/: one class number per pixel (0, 1, 2)
```

A frame's name is `<video name>_<frame index, 6 digits>`, e.g. `IMG_4884_000261`. A label file, in pixels of the
full-size frame:

```json
{
 "video": "data/input/IMG_4884.MOV", "frame_idx": 261, "image": "IMG_4884_000261.jpg", "width": 1920, "height": 1080,
 "status": "sam3",
 "water": [[[0, 352], [1740, 340], [1918, 366], [1918, 1079], [0, 1079]]],
 "ropes": [[[4.2, 487.5], [1915.9, 405.5], [1916.1, 410.5], [3.8, 496.5]]],
 "source": "sam3-litetext-s0"
}
```

`status`: `sam3` (not checked yet), `fixed` (changed by hand), `accepted` (checked, SAM3 was right). Training uses
only `fixed` and `accepted` frames.

Files written before 2026-11-16 store a rope as `{"points": [[x1, y1], [x2, y2]], "widths": [w1, w2]}`; it is read as
the 4 corners of that band (the same training mask, checked on all 182 files) and written as a polygon at the next
save.

## Steps

```bash
.venv/bin/python -m swim.labeling pick data/input/*.MOV   # 1. pick frames: one every 2 s, at most 25 per video
.venv/bin/python -m swim.labeling auto                     # 2. SAM3-LiteText labels them (about 2 s per frame)
.venv/bin/python -m swim.app                               # 3. check and fix them: http://127.0.0.1:8000/labels
.venv/bin/python -m swim.labeling masks                    # 4. training masks from the checked frames
.venv/bin/python -m swim.labeling status                   # how many frames are checked
```

- `pick` skips frames already in the dataset, so it can be run again with more videos or `--every 1`.
- `auto` labels only frames without SAM3 labels (`--redo`: all again). It never replaces labels you checked.
- `masks` writes masks for the checked frames only (`--all`: also the unchecked ones) and removes masks of frames no
  longer checked, so `masks/` is exactly the training set.
- Start with about 100 frames, train, look where the model fails, then add frames like those.

This is training-data tooling, not video processing: it has its own command (`python -m swim.labeling`), not
`run_pipeline()`.

## How SAM3's labels are made (`swim/labeling/sam3_labeler.py`, `shapes.py`)

Each frame is encoded once, then four prompts: "swimming pool", "lane rope", "blue stripe" (floor lane lines),
"person".

- Water: the "swimming pool" mask plus the ropes and the floor lines (SAM3 leaves the dark floor lines out of the pool
  mask; where they reach the frame's edge they would cut notches into the outline), plus each person whose
  surroundings are mostly pool (a swimmer on the pool's edge; SAM3 leaves swimmers out too), holes filled (swimmers
  inside the pool), gaps thinner than 2% of the frame closed (a rope the masks don't fully cover would split the
  pool), notches that reach the frame's edge filled (along each edge, non-pool stretches shorter than 20% of the edge
  between two pool stretches: a swimmer or a floor marking running out of the picture; stretches touching a corner,
  usually deck, are left alone), pieces under 0.5% of the frame dropped, then simplified to a polygon (8 px
  tolerance), with corners near the frame's edge moved onto it and the corners in between along that edge dropped.
- Ropes: a straight line through each rope mask (robust fit), its ends at the mask's extent, the width near each end
  from the mask; masks that are not line-shaped (length under 5 × width) are dropped; pieces on one line (within 3°
  and a few pixels) are joined, e.g. a rope cut by a swimmer. Saved as the 4 corners of that band.

## The Labels page (`/labels`)

| Part | What it does |
|---|---|
| Frame list | Every frame with its status (orange: not checked, green: checked), filter (all / not checked / fixed / accepted), how many are checked. Click to open |
| Editor | The frame with the water (blue) and ropes (orange) drawn on top. A tooltip next to the mouse names the class under it, as the training mask will have it (Lane rope 2, Water, Other; over a corner also what dragging it does). Corners: squares for water, circles for ropes; drag one to move it. Click a rope or water area to select it; the selected area shows + handles on its edges to add a corner. Drag on the frame to move it, mouse wheel to zoom |
| Tools | + Rope (R), + Water (W), − Water (E): click dots around the area. Each dot connects to the last, and from the third dot on the area fills in (the closing edge dashed); nothing follows the mouse. A drag moves the frame meanwhile (a click without moving is a dot). Click the first dot, double-click or press Enter to finish; Backspace (or Z) removes the last dot, Esc cancels. A click beyond the frame puts the dot on its edge. A new rope is added as drawn; water is merged into the water or cut out of it, and the outlines are simplified again on the server (the fast fix for a swimmer cut out of the pool, or deck marked as water). Join ropes (J: select a rope, click the other: one outline around both), Delete (the selected corner, rope or water area), Undo / Redo (Z / Shift+Z), Show labels (H), fill opacity |
| Save | Save (S) and Save & next (A). The status is set from the shapes: `fixed` if they differ from SAM3's, `accepted` if not. ← / → open the previous / next frame and save changes first. "Back to SAM3's labels" restores the model's shapes (undoable) |

API (`swim/app/server.py`): `GET /api/labels` (frames and statuses), `GET /api/labels/<name>` (labels and SAM3's),
`GET /api/labels/<name>/image`, `PUT /api/labels/<name>` (status, water, ropes; checked against the label format; the
frame's video, index and size come from the dataset, not the page), `POST /api/labels/<name>/water` (+ Water /
− Water: the water polygons with an area merged in or cut out; nothing saved).

## Next

- Train the model on `masks/` (architecture and size to decide; it must run fast on a Mac on every frame).
- Use it in a `pool` stage of the pipeline: water per frame (drop boxes outside the pool), ropes as lines per frame
  (lane of each box, for joining broken tracks).
