# Reading List

Papers to read for this project. Mark `[x]` when done and add a short note.

## Person detection

- [ ] **RF-DETR: Neural Architecture Search for Real-Time Detection Transformers** (2025)
  Isaac Robinson, Peter Robicheaux, Matvei Popov, Deva Ramanan, Neehar Peri (Roboflow, Carnegie Mellon University)
  [arXiv:2511.09554](https://arxiv.org/abs/2511.09554) · [PDF](https://arxiv.org/pdf/2511.09554) · [Code](https://github.com/roboflow/rf-detr) · [Blog](https://blog.roboflow.com/rf-detr/)
  _Why:_ our current best person detector: found the swimmer in 90/93 frames vs 41–60 for YOLOX on `IMG_4886.MOV`.
  Real-time DETR with a DINOv2 backbone; uses neural architecture search to pick the accuracy/speed trade-off.
  Designed for fine-tuning, which we will likely need for swimmers.

- [ ] **Detecting Swimmers in Unconstrained Videos with Few Training Data** (2021)
  [ResearchGate](https://www.researchgate.net/publication/355033619_Detecting_Swimmers_in_Unconstrained_Videos_with_Few_Training_Data)
  _Why:_ swimmer-specific detection; introduces the Swimm400 dataset.

## Pose estimation (keypoints)

- [ ] **RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose** (2023)
  [arXiv:2303.07399](https://arxiv.org/abs/2303.07399)
  _Why:_ the pose model we use via `rtmlib`. Top-down: depends on the person detector's box.

- [ ] **SwimmerNET: Underwater 2D Swimmer Pose Estimation Exploiting Fully Convolutional Neural Networks** (2023)
  [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9966167/)
  _Why:_ swimmer-specific keypoint detection from a single wide-angle camera.

- [ ] **RTMPose and Ensemble Learning for Real-Time Swimmer Talent Detection** (2026)
  [Springer](https://link.springer.com/chapter/10.1007/978-3-032-07336-5_17)
  _Why:_ RTMPose applied to swimmers with an underwater-view keypoint dataset.

## Datasets

- [ ] **SwimXYZ: A large-scale dataset of synthetic swimming motions and videos** (2023)
  [arXiv:2310.04360](https://arxiv.org/abs/2310.04360)
  _Why:_ 11,520 synthetic swim videos with 2D/3D keypoints; candidate data for fine-tuning.

## Background (optional)

- [ ] **DETR: End-to-End Object Detection with Transformers** (2020)
  [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
  _Why:_ the original detection transformer that RF-DETR builds on.

- [ ] **DINOv2: Learning Robust Visual Features without Supervision** (2023)
  [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
  _Why:_ the backbone inside RF-DETR; explains why it generalizes well to unusual scenes like pools.
