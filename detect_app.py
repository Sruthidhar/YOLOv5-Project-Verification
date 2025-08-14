import torch
import streamlit as st
from PIL import Image
import numpy as np

# Title
st.title("👀 People Detector using YOLOv5")
st.write("Upload an image, and I’ll detect and count people in it.")

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.conf = 0.25
    model.classes = [0]  # Only detect people
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run detection
    results = model(img)
    df = results.pandas().xyxy[0]
    count = len(df)

    # Render results
    results.render()
    detected_img = Image.fromarray(results.ims[0])

    st.image(detected_img, caption=f"Detected People: {count}", use_column_width=True)
    st.success(f"✅ Total People Counted: {count}")
