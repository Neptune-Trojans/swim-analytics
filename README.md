# swim-analytics

Process a swimming video and overlay analysis layers (skeleton, joint angles, velocity, speed) on it.

## Structure

```
swim/
  main.py             # entry point: input video -> output video
  video.py            # read frames / write output video
  detections.py       # detection data classes (BBox, Detection, ...) + Detector contract
  detectors/          # one file per detection model
    rfdetr_detector.py
  pose.py             # pose model -> keypoints per frame
  metrics.py          # joint angles, velocity, speed
  draw.py             # draw overlays on frames
scripts/              # quick check scripts (run a model on a video)
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
