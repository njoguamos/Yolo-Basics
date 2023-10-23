from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Train the model using the 'config.yaml' dataset for 2 epochs
model.train(data="config.yaml", epochs=4)