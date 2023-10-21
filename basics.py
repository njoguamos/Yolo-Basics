from ultralytics import YOLO
import cv2

model = YOLO('./Yolo Weights/yolov8x.pt')

results = model('./images/cars.jpg', show=True)

# Unless user input, don't do anything`
cv2.waitKey(0)