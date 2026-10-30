"""Train YOLOv11s for fire and smoke detection.

Settings follow the thesis (Table 3.3): image size 640, batch 16, 100 epochs,
AdamW optimizer, initial learning rate 0.001, mosaic augmentation.

Usage:
    python train.py                      # full 100-epoch run, as in the thesis
    python train.py --epochs 20          # shorter run (about 6 min/epoch on an RTX 3050)
    python train.py --batch 8            # if you run out of GPU memory
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11s on the fire & smoke dataset")
    parser.add_argument("--data", default="data.yaml", help="path to dataset config")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="GPU id, or 'cpu'")
    args = parser.parse_args()

    # Start from COCO-pretrained YOLOv11s weights (downloaded automatically) and fine-tune.
    model = YOLO("yolo11s.pt")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=0.001,
        mosaic=1.0,      # mosaic augmentation (4 images combined into 1 training sample)
        hsv_h=0.015,     # HSV colour augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,      # horizontal flip
        translate=0.1,   # translation
        scale=0.5,       # random scaling
        device=args.device,
        project="runs",
        name="fire_smoke_yolo11s",
    )
    print("Done. Best weights: runs/fire_smoke_yolo11s/weights/best.pt")
    print("Copy best.pt into the weights/ folder to use it in the app.")


if __name__ == "__main__":
    main()
