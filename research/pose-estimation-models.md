# Pose Estimation Models for Swimming

_Research date: 2026-10-05_

Goal: find neural networks that detect skeleton joints (keypoints) from swimming video.

## Key challenge

Off-the-shelf pose models are trained mostly on people on land (COCO). Swimming is harder:

- Splashes, bubbles and the water surface hide arms and legs.
- Refraction distorts limbs that are partly underwater.
- The body is horizontal instead of upright.

Expect to test several models on our own clips before committing to one.

## Candidate models

| Model | Joints | Speed / accuracy | Install | License | Fit for us |
|---|---|---|---|---|---|
| **RTMPose** (via `rtmlib`) | 17 body, or 133 whole-body (RTMW) | RTMPose-m: 75.8% COCO AP, 90+ FPS on Intel i7 CPU | `pip install rtmlib` (ONNX, no mmcv) | Apache-2.0 | **Best starting choice.** Fast, accurate, used in 2026 swimmer research |
| **YOLO pose** (Ultralytics, YOLO26-Pose) | 17 | Fastest; person detection + keypoints in one pass | `pip install ultralytics` | **AGPL-3.0** (commercial license needed for closed source) | Easiest to try and to fine-tune on our own labels |
| **MediaPipe Pose** | 33 (incl. feet, hands) | Fast on CPU | `pip install mediapipe` | Apache-2.0 | Good for a quick demo; tuned for upright, clearly visible people, often loses swimmers |
| **ViTPose / ViTPose++** | 17+ | Most accurate, slow (large transformer) | MMPose or Hugging Face | Apache-2.0 | Offline high-accuracy runs, or generating draft labels |

## Swimming-specific research

- **SwimmerNET** (2023): underwater 2D swimmer pose estimation with fully convolutional networks and a single wide-angle camera.
- **RTMPose for swimmer talent detection** (2026): compares MMPose models and RTMPose on keypoints labelled from an underwater view.
- **SwimXYZ**: synthetic dataset, 11,520 videos with 2D/3D keypoints and SMPL parameters across four strokes. Useful for fine-tuning.
- **YOLOv7 Swim Pose Recognition**: open-source project that fine-tunes YOLOv7 pose with augmented swim footage.
- Real underwater swimmer datasets with keypoints are rare and mostly not fully public.

## Recommendation

1. Start with **RTMPose via `rtmlib`**: accurate, fast, simple install, permissive license.
2. Compare against **YOLO pose** and **MediaPipe** on the same clip; measure how often limbs are lost.
3. Keep the model behind one interface in `swim/pose.py` (e.g. `detect(frame) -> keypoints`) so swapping is trivial.
4. If accuracy is not good enough: label a few hundred of our own frames and fine-tune YOLO pose or RTMPose, possibly adding SwimXYZ.

**Camera angle:** an above-water side view is the hardest case. An underwater side view gives much better results and is what most swimming research uses.

## Sources

- [RTMPose and Ensemble Learning for Real-Time Swimmer Talent Detection (2026)](https://link.springer.com/chapter/10.1007/978-3-032-07336-5_17)
- [SwimmerNET: Underwater 2D Swimmer Pose Estimation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9966167/)
- [SwimXYZ synthetic swimming dataset](https://arxiv.org/html/2310.04360)
- [YOLOv7 Swim Pose Recognition (GitHub)](https://github.com/JonOuyang/YOLOv7-Swim-Pose-Recognition)
- [3D Human Pose Estimation in Underwater Scenarios](https://doi.org/10.3390/s26154738)
- [Pose Estimation of Swimmers Using Deep Learning (AUT)](https://openrepository.aut.ac.nz/server/api/core/bitstreams/6f5dc283-2060-4ee7-a392-03e19ee2523a/content)
- [Ultralytics YOLO26 paper](https://arxiv.org/pdf/2606.03748)
- [RTMW: Real-Time Whole-body Pose Estimation](https://arxiv.org/pdf/2407.08634)
- [Pose Estimation 2026 overview (Label Your Data)](https://labelyourdata.com/articles/data-annotation/pose-estimation)
- [Best Pose Estimation Models (Roboflow)](https://blog.roboflow.com/best-pose-estimation-models/)
- [rtmpose GitHub topic](https://github.com/topics/rtmpose)
