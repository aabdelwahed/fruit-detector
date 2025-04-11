import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import requests
from PIL import Image, ImageDraw
import numpy as np
import io

st.title("📱 Browser Camera - Object Detection")

PREDICTION_URL = "https://bigcatprediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/391dd797-5d5a-4ceb-a2bb-dd24fddabb66/detect/iterations/Iteration1/image"
PREDICTION_KEY = "5eTkKKFHbjhbC2DKmb3XvcKlQFaEieGcbisaKsH1AW37UemBTzH5JQQJ99BDACLArgHXJ3w3AAAIACOGCFlW"

headers = {
    "Prediction-Key": PREDICTION_KEY,
    "Content-Type": "application/octet-stream"
}

def detect_objects(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    response = requests.post(PREDICTION_URL, headers=headers, data=image_bytes)
    return response.json()

def draw_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for pred in predictions:
        tag = pred["tagName"]
        prob = pred["probability"]
        bbox = pred["boundingBox"]
        left = bbox["left"] * w
        top = bbox["top"] * h
        width = bbox["width"] * w
        height = bbox["height"] * h
        draw.rectangle([left, top, left + width, top + height], outline="red", width=3)
        draw.text((left, top), f"{tag} ({prob*100:.1f}%)", fill="white")
    return image

class VideoProcessor:
    def recv(self, frame):
        image = frame.to_image()
        result = detect_objects(image)
        predictions = result.get("predictions", [])
        image = draw_boxes(image, predictions)
        return av.VideoFrame.from_image(image)

webrtc_streamer(key="fruit-detect", video_processor_factory=VideoProcessor)

