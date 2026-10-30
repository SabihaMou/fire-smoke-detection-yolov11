"""Streamlit GUI for real-time fire and smoke detection with YOLOv11.

Three input modes: image upload, video upload, live webcam.
Each result shows bounding boxes, class labels and confidence scores.

Run:
    streamlit run app.py
Put your trained weights at weights/best.pt (or set env var FIRE_MODEL=path/to/best.pt).
"""
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.environ.get("FIRE_MODEL", "weights/best.pt")

st.set_page_config(page_title="Fire & Smoke Detection System", page_icon="🔥", layout="wide")


@st.cache_resource
def load_model(path: str) -> YOLO:
    """Load the model once and keep it in memory between reruns."""
    return YOLO(path)


def detect(model: YOLO, frame_bgr: np.ndarray, conf: float):
    """Run detection on one BGR frame. Returns (annotated RGB image, YOLO result)."""
    result = model.predict(frame_bgr, conf=conf, imgsz=640, verbose=False)[0]
    annotated_bgr = result.plot()  # draws boxes, labels, confidences
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), result


def detections_table(result) -> pd.DataFrame:
    rows = [
        {"class": result.names[int(b.cls)], "confidence": round(float(b.conf), 2)}
        for b in result.boxes
    ]
    return pd.DataFrame(rows, columns=["class", "confidence"])


# ---------------------------------------------------------------- page header
st.title("🔥 Fire and Smoke Detection System")
st.caption("YOLOv11-based Real-Time Detection System")

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model weights not found at '{MODEL_PATH}'. Train a model with `python train.py` "
        "and copy runs/fire_smoke_yolo11s/weights/best.pt to weights/best.pt."
    )
    st.stop()

model = load_model(MODEL_PATH)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Detection Mode")
mode = st.sidebar.radio("Select Input Type", ["Image Detection", "Video Detection", "Webcam Detection"])
conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)

# ---------------------------------------------------------------- image mode
if mode == "Image Detection":
    st.header("🖼️ Image Detection")
    uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if st.button("Run Detection"):
            annotated, result = detect(model, frame_bgr, conf)
            st.image(annotated, caption="Detection Result", use_container_width=True)
            if len(result.boxes):
                st.dataframe(detections_table(result), use_container_width=True)
            else:
                st.info("No fire or smoke detected at this confidence threshold.")

# ---------------------------------------------------------------- video mode
elif mode == "Video Detection":
    st.header("🎬 Video Detection")
    uploaded = st.file_uploader("Upload a Video", type=["mp4", "avi"])
    if uploaded is not None and st.button("Run Video Detection"):
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        placeholder = st.empty()
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, _ = detect(model, frame, conf)  # each frame is detected independently
                placeholder.image(annotated, use_container_width=True)
        finally:
            cap.release()
            os.remove(video_path)
        st.success("Video finished.")

# ---------------------------------------------------------------- webcam mode
else:
    st.header("📷 Webcam Detection")
    st.write("Click **Start Webcam**. To stop, click any other control (this interrupts the loop).")
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open the webcam.")
        else:
            placeholder = st.empty()
            try:
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    annotated, _ = detect(model, frame, conf)
                    placeholder.image(annotated, use_container_width=True)
            finally:
                cap.release()
