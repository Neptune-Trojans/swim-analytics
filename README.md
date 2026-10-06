# swim-analytics

Process a swimming video and overlay analysis layers (skeleton, joint angles, velocity, speed) on it.

## Structure

```
swim/
  main.py             # entry point: input video -> output video
  video.py            # video helpers: read frames, write video, output paths
  detections.py       # detection data classes (BBox, Detection, ...) + Detector contract
  detectors/          # one file per detection model
    rfdetr_detector.py
  visualization/      # all drawing code (one file per data type)
    detections.py     # draw detection boxes on frames
  pose.py             # pose model -> keypoints per frame
  metrics.py          # joint angles, velocity, speed
scripts/              # quick check scripts
  run_rfdetr.py             # video -> detections JSON + annotated video
  visualize_detections.py   # video + detections JSON -> annotated video (no model)
  run_rtmlib.py             # video -> skeleton video (RTMPose)
data/
  input/              # source swim videos
  output/             # processed videos
notebooks/            # experiments
research/             # research notes (one .md per topic)
```

## Usage

```bash
pip install -e .
python -m swim.main data/input/swim.mp4 -o data/output/swim.mp4
```
