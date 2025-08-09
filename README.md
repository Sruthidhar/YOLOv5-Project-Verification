# YOLOv5 Project Verification

This repository contains code and scripts provided for **YOLOv5 model verification**.  
The purpose is to test, export, and verify object detection functionality for project review.

---

## 📂 Contents
- **`export.py`** → Exports YOLOv5 `.pt` models to ONNX, TorchScript, and CoreML formats.
- **`experimental.py`** → Contains experimental model modules (GhostConv, CrossConv, etc.).
- **`test_cam.py`** → Simple OpenCV script to verify webcam functionality.
- **YOLOv5 Utility Files** → Supporting modules from the YOLOv5 framework.

---

## ⚙️ Requirements
Install Python dependencies before running:
```bash
pip install torch torchvision onnx coremltools opencv-python numpy
