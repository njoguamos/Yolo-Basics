from ultralytics import YOLO
import cv2
import cvzone
import math

# Initialise video capture
capture = cv2.VideoCapture(0)

# Set camera dimensions
capture.set(3, 1280)
capture.set(4, 720)

# load a pretrained model
model = YOLO('./Yolo Weights/yolov8x.pt')

classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

while True:
    # reads the next frame from the video source and stores the success
    # if successful, the image data is stored in the img variable.
    success, img = capture.read()
    # predict on an image and stream the results
    results = model(img, stream=True)
    # Loop through the object detected
    for r in results:
        # loop through the bounding boxes
        boxes = r.boxes
        for box in boxes:
            # Get the boundix box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            # Convert the coordinated to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Compute height and width of the bounding box
            w, h = x2 - x1, y2 - y1
            # Draw a rectangle on the images
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Class name
            cls = int(box.cls[0])
            # Get the confidence to 2dp
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Show a rectangle with confidence
            cvzone.putTextRect(img, f'{conf} {classNames[cls]}', (max(0, x1), max(35, y1)), scale=0.96, thickness=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
