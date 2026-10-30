# Fire and Smoke Detection System based on YOLOv11

Real-time fire and smoke detection from images, video files and a live webcam, built with
**YOLOv11s** (Ultralytics, PyTorch) and a **Streamlit** interface.

This is my undergraduate thesis project (Bachelor of Engineering in Computer Science and Technology,
Wuhan Institute of Technology, 2026; supervisor: Assoc. Prof. Xu Cheng).
Thesis title: *Design and Implementation of a Fire Detection System Based on YOLOv11*.

> **About this repository (please read).** The original working files of my thesis experiments
> (trained weights and notebooks) were not kept after I submitted the thesis. This repository is a clean
> re-implementation of the same pipeline (training, evaluation and the Streamlit app) using the settings
> reported in the thesis. The **result figures and numbers below come from the experiments reported in my thesis**
> (RTX 3050 4 GB GPU). Trained weights are not included; run `train.py` to reproduce them.
> The code for the three comparison models (Faster R-CNN, RetinaNet, EfficientDet) is not part of this repository.

## What it does

- Detects **fire** and **smoke** (and an `other` class of fire-like distractors such as sunlight, fog, bright lights) with bounding boxes, labels and confidence scores
- Three input modes: **image**, **video**, **live webcam**
- Compares four detectors in the thesis: YOLOv11s, Faster R-CNN, RetinaNet, EfficientDet

## Dataset

[Fire & Smoke Dataset | Object Detection | YOLO](https://www.kaggle.com/datasets/azimjaan21/fire-and-smoke-dataset-object-detection-yolo)
(Kaggle, more than 17,000 annotated images from forests, buildings, industrial areas and CCTV scenes, in YOLO format).
The dataset belongs to its original authors and is not redistributed here.

## Results (from the thesis)

| Model | Precision | Recall | mAP@50 | mAP@50:95 | FPS |
|---|---|---|---|---|---|
| **YOLOv11s** | 0.90 | 0.89 | **0.93** | 0.58 | **42** |
| Faster R-CNN | 0.90 | 0.84 | 0.90 | 0.57 | 11 |
| RetinaNet | 0.86 | 0.81 | 0.85 | 0.54 | 24 |
| EfficientDet | 0.85 | 0.83 | 0.85 | 0.56 | 28 |

YOLOv11s gave the best accuracy and the fastest inference (23.8 ms per image), so it was chosen for the application.
Fire (AP@50 about 0.95) was detected more reliably than smoke (about 0.89), because smoke is transparent, irregular and easily confused with fog and clouds.

<p align="center"><img src="docs/figures/training_curves.png" width="800"><br><em>YOLOv11s training curves (100 epochs)</em></p>

<p align="center">
<img src="docs/figures/confusion_matrix.png" width="380">
<img src="docs/figures/pr_curve.png" width="380"><br>
<em>Confusion matrix and precision-recall curve (YOLOv11s)</em></p>

## Application screenshots

<p align="center">
<img src="docs/figures/gui_image_detection.png" width="380">
<img src="docs/figures/gui_video_detection.png" width="380">
</p>

## Quick start

```bash
git clone https://github.com/<your-username>/fire-smoke-detection-yolov11.git
cd fire-smoke-detection-yolov11
pip install -r requirements.txt
```

1. Download the dataset from Kaggle into `datasets/fire_smoke/` and check that `data.yaml` matches its folders and class order.
2. Train (about 6 minutes per epoch on an RTX 3050; use `--epochs 20` for a quick run):
   ```bash
   python train.py
   ```
3. Copy `runs/fire_smoke_yolo11s/weights/best.pt` to `weights/best.pt`.
4. Evaluate: `python evaluate.py --weights weights/best.pt`
5. Launch the app: `streamlit run app.py`

## Training settings (thesis, Table 3.3)

Image size 640x640 · batch size 16 · 100 epochs · AdamW · learning rate 0.001 · loss: DFL + CIoU (+ classification loss) ·
augmentation: mosaic, HSV colour, flip, scale, translation.

## Project structure

```
app.py          Streamlit interface (image / video / webcam)
train.py        training script
evaluate.py     precision, recall, mAP on the test split
data.yaml       dataset configuration
docs/           figures from the thesis, learning notes
```

## Limitations and future work

Smoke is harder than fire; fog, clouds and strong lighting can cause false alarms; the dataset has little night-time,
heavy-rain and aerial wildfire data; only RGB input is used. Future directions: more diverse data, thermal/infrared fusion,
transformer-based detectors (e.g. RT-DETR), use of temporal information across video frames, and lightweight models for edge devices.

## Author

Mou Sabiha Sultana · sabihamou111@gmail.com
