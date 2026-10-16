from ultralytics import YOLO

# Load a small pretrained object-detection model.
model = YOLO("yolo26n.pt")

print("YOLO loaded successfully.")
print("Available classes:")
print(model.names)