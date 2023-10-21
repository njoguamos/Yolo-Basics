import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math

from sort.sort import Sort

# Initialise video capture
capture = cv2.VideoCapture('./videos/cars.mp4')

# Image to mask the video
mask = cv2.imread('./videos/mask.png')

# tracker
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

# limit
limits = [156, 678, 1120, 678]

# total vehicle counts
total_count = []

# load a pretrained model
model = YOLO('./Yolo Weights/yolov8n.pt')

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
    img_region = cv2.bitwise_and(src1=img, src2=mask)
    # predict on an image and stream the results
    results = model(img_region, stream=True, device="mps")
    # Initialise an array for detections
    detections = np.empty((0, 5))
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

            # Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            # Get the confidence to 2dp
            conf = math.ceil((box.conf[0] * 100)) / 100

            if currentClass in ["car", "motorcycle", 'bus', 'truck'] and conf > 0.3:
                # Draw a rectangle on the images
                cvzone.cornerRect(img, bbox=(x1, y1, w, h), l=9, rt=5)

                # Show a rectangle with confidence
                # cvzone.putTextRect(img, f'{conf} {currentClass}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1,
                #                    offset=8)
                # Update detections
                current_array = np.array([x1, y1, x2, y2, conf])
                # stack the detections (
                detections = np.vstack((detections, current_array))

    results_tracker = tracker.update(dets=detections)

    cv2.line(img, pt1=(limits[0], limits[1]), pt2=(limits[2], limits[3]), color=(0, 0, 255), thickness=3)

    # show tracker bounding box
    for res in results_tracker:
        x1, y1, x2, y2, Id = res
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        # Find center points
        cx, cy = x1 + w / 2, y1 + h / 2
        # Draw the center points
        cv2.circle(img, center=(int(cx), int(cy)), radius=5, color=(255, 0, 255), thickness=cv2.FILLED)

        # Create a region to detect the point
        if limits[0] < cx < limits[2] and limits[1] - 30 < cy < limits[1] + 30:
            if total_count.count(Id) == 0:
                total_count.append(Id)

    cvzone.putTextRect(img, f'Total Count: {len(total_count)}', pos=(50, 50))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
