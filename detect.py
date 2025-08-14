import torch
import cv2
import sys
import os

# Check if image path was provided
if len(sys.argv) < 2:
    print("❌ Usage: python detect.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Check if file exists
if not os.path.isfile(image_path):
    print(f"❌ File not found: {image_path}")
    sys.exit(1)

# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Only detect persons (class 0 in COCO dataset)
model.classes = [0]

# Inference
results = model(image_path)

# Results to image (with bounding boxes)
results.render()

# Display the image
for img in results.ims:
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Person Detection", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Save the output
results.save(save_dir="output")

print("✅ Detection complete. Results saved in 'output' folder.")
