# Person Detection for Swimming

_Research date: 2026-10-06_

Goal: reliably find the swimmer (bounding box) in every frame **before** working on keypoint detection.
Top-down pose models (like RTMPose) only work as well as the box they are given.

## Why start here

Findings from testing `rtmlib` (YOLOX detector + RTMPose) on `IMG_4886.MOV` (93 frames, backstroke, above-water view):

| Mode | Detector | Frames where the swimmer was found |
|---|---|---|
| `balanced` | YOLOX-m | 41 / 93 (44%) |
| `performance` | YOLOX-x | 60 / 93 (65%) |

- **When the swimmer is found**, the RTMPose skeleton is mostly reasonable (head, shoulders, arms, legs roughly right).
- **When the swimmer is missed**, `rtmlib` silently falls back to using the whole frame as the box, and the pose model
  draws a skeleton in random places (e.g. on the dark T-mark on the pool floor).
- **Lowering the detector confidence threshold** (0.7 → 0.5 → 0.3 → 0.2) changed nothing: same frames detected.
  This is a model limitation, not a setting to tune.

Conclusion: the detector is the bottleneck, not the pose model.

## Plan

### Step 1: Build a small test set from our own videos

- Extract every 10th frame from the videos in `data/input/` (~100–200 frames, covering different strokes and camera angles).
- Draw a box around the swimmer by hand using a free tool: [CVAT](https://www.cvat.ai/) or [Label Studio](https://labelstud.io/).
- Export in YOLO or COCO format. Roughly 1–2 hours of work.
- Tip: an open-vocabulary model (Grounding DINO / OWLv2 with the prompt "swimmer") can pre-draw boxes so we only correct them.

### Step 2: Score each candidate on that set

For each model, measure:

- **Recall**: % of frames where the swimmer is found (the most important).
- **False positives**: boxes on lane lines, pool marks, splashes.
- **Box fit**: IoU with the hand-drawn box.
- **Speed**: fps on the Mac (CPU).

### Step 3: Pick the winner, add tracking, then return to keypoints

- Add a tracker (or simply carry the last box forward) to fill frames the detector misses. There is one swimmer
  who moves smoothly, so this should cover most gaps.
- Feed the boxes into the pose model (RTMPose or others) and evaluate keypoints.

### Step 4 (if needed): Fine-tune

If no off-the-shelf model is good enough, fine-tune the best general detector on our labelled frames plus public
swimmer datasets (below). This is what most swimming-specific research does.

## Candidate models

| Model | Type | Why try it | License |
|---|---|---|---|
| **YOLOX** (current, via `rtmlib`) | General detector | Baseline: 44% / 65% recall on our clip | Apache-2.0 |
| **Ultralytics YOLO26 / YOLO11** | General detector | Newest YOLO, fast on CPU, very easy to fine-tune | AGPL-3.0 (commercial license needed for closed source) |
| **RF-DETR** (Roboflow) | Transformer detector | More accurate than YOLO in published benchmarks, easy to fine-tune | Apache-2.0 |
| **Grounding DINO / OWLv2** | Open-vocabulary (text prompt "swimmer") | Needs no training; slow, but shows what's possible and helps auto-label | Apache-2.0 |
| **YOLO fine-tuned on swimmer datasets** | Swimming-specific | Trained on swimmers in pools | Varies |

## Swimming-specific datasets

- **Roboflow Universe swimmer dataset**: ready-to-use swimmer boxes.
- **Swimm400**: swimmer detection dataset from "Detecting Swimmers in Unconstrained Videos with Few Training Data".
- **Pool drowning-detection datasets**: e.g. 12,365 outdoor pool images labelled swimming / drowning / out of water.
- **Underwater Drowning Detection Dataset**: 5,613 annotated underwater images.

## Sources

- [Roboflow Universe swimmer dataset](https://universe.roboflow.com/yolo-pdvpx/swimmer-caayd)
- [Detecting Swimmers in Unconstrained Videos with Few Training Data](https://www.researchgate.net/publication/355033619_Detecting_Swimmers_in_Unconstrained_Videos_with_Few_Training_Data)
- [A Pool Drowning Detection Model Based on Improved YOLO](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12431139/)
- [Underwater Drowning Detection Dataset (figshare)](https://figshare.com/articles/dataset/Underwater_Drowning_Detection_Dataset/29497235)
- [Swimming-YOLO: drowning detection in multi-swimming scenarios](https://www.researchgate.net/publication/387496674_Swimming-YOLO_a_drowning_detection_method_in_multi-swimming_scenarios_based_on_improved_YOLO_algorithm)
- [RF-DETR vs. Alternatives: Benchmarks and Deployment](https://roboflow.com/blog/rf-detr-vs-alternatives)
- [Real-Time Detection 2026: RF-DETR vs YOLO26 vs SAM 3](https://builderai.tools/blog/real-time-object-detection-rf-detr-yolo-sam-3)
- [Open-Vocabulary Detection: Grounding DINO, Florence-2, OWLv2, RT-DETR, RF-DETR](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/open-vocabulary-detection-grounding-dino-florence-2-rtdetr-rfdetr)
- [Ultralytics YOLO26 paper](https://arxiv.org/pdf/2606.03748)
