import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import requests
from PIL import Image, ImageDraw
from collections import Counter
from datetime import datetime
import io
import os

st.set_page_config(page_title="Fruit Detector", layout="centered")
st.title("🍎 Real-Time Fruit Detection with Snapshot")

# === Azure Custom Vision Info ===
PREDICTION_URL = "https://bigcatprediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/391dd797-5d5a-4ceb-a2bb-dd24fddabb66/detect/iterations/Iteration1/image"
PREDICTION_KEY = "5eTkKKFHbjhbC2DKmb3XvcKlQFaEieGcbisaKsH1AW37UemBTzH5JQQJ99BDACLArgHXJ3w3AAAIACOGCFlW"

headers = {
    "Prediction-Key": PREDICTION_KEY,
    "Content-Type": "application/octet-stream"
}

# === Detection Function ===
def detect_objects(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    response = requests.post(PREDICTION_URL, headers=headers, data=buffered.getvalue())
    return response.json()

# === Draw Boxes Function ===
def draw_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for pred in predictions:
        if pred["probability"] >= 0.4:
            tag = pred["tagName"]
            prob = pred["probability"]
            bbox = pred["boundingBox"]
            left = bbox["left"] * w
            top = bbox["top"] * h
            width = bbox["width"] * w
            height = bbox["height"] * h
            draw.rectangle([left, top, left + width, top + height], outline="red", width=3)
            draw.text((left, top - 10), f"{tag} ({prob*100:.1f}%)", fill="white")
    return image

# === Snapshot Save ===
def save_snapshot(image, predictions):
    os.makedirs("snapshots", exist_ok=True)
    tag_names = [p["tagName"] for p in predictions]
    tags = "_".join(sorted(set(tag_names)))
    filename = f"snapshots/{tags}_{datetime.now().strftime('%H%M%S')}.jpg"
    image.save(filename)
    return filename

# === Video Processing ===
class VideoProcessor:
    def __init__(self):
        self.last_image = None
        self.last_predictions = []

    def recv(self, frame):
        image = frame.to_image()
        result = detect_objects(image)
        predictions = result.get("predictions", [])
        boxed_image = draw_boxes(image.copy(), predictions)

        self.last_image = boxed_image
        self.last_predictions = predictions

        return av.VideoFrame.from_image(boxed_image)

vp = VideoProcessor()
ctx = webrtc_streamer(key="realtime", video_processor_factory=lambda: vp)

# === Real-time Display ===
if vp.last_predictions:
    st.markdown("### 🧮 Object Count:")
    tag_counts = Counter([p["tagName"] for p in vp.last_predictions])
    for tag, count in tag_counts.items():
        st.write(f"**{tag}**: {count}")

# === Snapshot Button ===
if vp.last_image and st.button("📸 Save Snapshot"):
    filename = save_snapshot(vp.last_image, vp.last_predictions)
    st.success(f"Snapshot saved: {filename}")
