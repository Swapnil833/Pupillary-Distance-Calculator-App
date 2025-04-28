from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO("yolov11m-seg-custom.pt")

results = model.predict(source="1.jpg", show=True, save=True, conf=0.6)

for result in results:
    for mask in result.masks.xy:  
        polygon = np.array(mask, np.int32)  # Convert to integer
        polygon = polygon.reshape((-1, 1, 2))  

        # Get bounding rectangle (x, y, w, h)
        x, y, w, h = cv2.boundingRect(polygon)

        # Extract bottom-left coordinate
        bottom_left_coordinate = (x, y + h)
        
print("Bottom Left Coordinate:", bottom_left_coordinate)