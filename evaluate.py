"""Evaluate trained weights on the test split.

Reports precision, recall, mAP@50, mAP@50:95 and per-class AP@50,
using confidence threshold 0.25 and IoU threshold 0.50 (as in the thesis).

Usage:
    python evaluate.py --weights weights/best.pt
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv11 model")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, conf=0.25, iou=0.5)

    print("\n===== Results on the", args.split, "split =====")
    print(f"Precision : {m.box.mp:.3f}")
    print(f"Recall    : {m.box.mr:.3f}")
    print(f"mAP@50    : {m.box.map50:.3f}")
    print(f"mAP@50:95 : {m.box.map:.3f}")
    print("\nPer-class AP@50:")
    for idx, ap in zip(m.box.ap_class_index, m.box.ap50):
        print(f"  {m.names[int(idx)]:<8s} {ap:.3f}")


if __name__ == "__main__":
    main()
