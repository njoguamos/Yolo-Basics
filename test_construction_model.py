from ultralytics import YOLO
import cv2
import cvzone
import math

# Initialise video capture
capture = cv2.VideoCapture('./videos/ppes.mp4')

# load a pretrained model
model = YOLO('./runs/detect/train/weights/best.pt')

classNames = [
    'helmet', 'no-helmet', 'no-vest', 'person', 'vest'
]

while True:
    # reads the next frame from the video source and stores the success
    # if successful, the image data is stored in the img variable.
    success, img = capture.read()
    # predict on an image and stream the results
    results = model(img, stream=True, device="mps")
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
            # cvzone.putTextRect(img, f'{conf} {classNames[cls]}', (max(0, x1), max(2, y1)), scale=0.96, thickness=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
