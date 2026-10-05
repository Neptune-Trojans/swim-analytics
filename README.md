# swim-analytics

Process a swimming video and overlay analysis layers (skeleton, joint angles, velocity, speed) on it.

## Structure

```
swim/
  main.py     # entry point: input video -> output video
  video.py    # read frames / write output video
  pose.py     # pose model -> keypoints per frame
  metrics.py  # joint angles, velocity, speed
  draw.py     # draw overlays on frames
data/
  input/      # source swim videos
  output/     # processed videos
notebooks/    # experiments
```

## Usage

```bash
pip install -e .
python -m swim.main data/input/swim.mp4 -o data/output/swim.mp4
```
