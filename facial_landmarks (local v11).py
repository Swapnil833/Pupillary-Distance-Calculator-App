from imutils import face_utils, paths
from pupil_distance import *
import dlib
import cv2
import time
import os
from PIL import Image
import math
from ultralytics import YOLO
import shutil
import warnings
from glasses_detector import GlassesClassifier

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
warnings.simplefilter(action='ignore', category=FutureWarning)

model = YOLO("yolov11m-seg-custom.pt")
predictions_location = "D:\\I2I Techno Solutions\\I2I Techno Solutions\\Dlib_Static (Trial for YOLOv11)\\runs"
predictions_dir = "predict"
classifier = GlassesClassifier()

prediction_path = os.path.join(predictions_location, predictions_dir)
isExist = os.path.exists(prediction_path) 
if(isExist == True):
    shutil.rmtree(prediction_path)
else:
    pass

nose_bridge = []
x1 = y1 = x2 = y2 = 0
input_folder = input("Enter the path to the folder containing input images: ").strip()

for image_path in paths.list_images(input_folder):
    image_name = os.path.basename(image_path)
    #output_path = os.path.join(output_folder, image_name)
    image = cv2.imread(image_path)
    # load the image, resize it, and conver it to grayscale
    #image = cv2.imread("./images/image6.jpg")
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image 
    rects = detector(gray, 1)

    results = model.predict(source=image_path, show=True, save=True, conf=0.6, project='D:\\I2I Techno Solutions\\I2I Techno Solutions\\Dlib_Static (Trial for YOLOv11)\\runs')

    classifier.process_file(
        input_path=image_path,
        output_path="D:\\I2I Techno Solutions\\I2I Techno Solutions\\Dlib_Static (Trial for YOLOv11)\\runs\\predict",
        #\\runs\\detect\\predict  OR   output
        format={True: "1", False: "0"},
        show=True,
    )
   
    # Open the file in read mode
    text_file_location = "D:\\I2I Techno Solutions\\I2I Techno Solutions\\Dlib_Static (Trial for YOLOv11)\\runs\\predict"
    text_file_name = os.path.splitext(image_name)[0] + ".txt"
    text_file_path = os.path.join(text_file_location, text_file_name)
    
    if os.path.exists(text_file_path):
        with open(text_file_path, "r") as file:
            data_txt = file.read()
            print("Text File Content:\n", data_txt)
    else:
        print("Error: Text file not found:", text_file_path)
        
    content = int(data_txt)
    print("txt file content is:", content)
    file.close()

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to an OpenCV-style bounding box [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, 'Face #{}'.format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
        for (j, (x, y)) in enumerate(shape):
            if j in [37, 38, 40, 41, 43, 44, 46, 47]:
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
            else:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        # estimating center from known points 
        left_pupil = Pupil(shape[37], shape[40], shape[41], shape[38]).central_point
        right_pupil = Pupil(shape[43], shape[46], shape[47], shape[44]).central_point
    
        left_eyelid = Eyelid(shape[40], shape[41]).midpoint
        right_eyelid = Eyelid(shape[46], shape[47]).midpoint
    
        #Left Bifocal Length
        left_pupil_eyelid = ((((left_pupil.x - left_eyelid.x)**2) + ((left_pupil.y - left_eyelid.y + 10)**2))**0.5)
        real_distance_eyelid_left = (left_pupil_eyelid)/4.3
    
        cv2.line(image,(int(left_pupil.x), int(left_pupil.y)), (int(left_eyelid.x), (int(left_eyelid.y) + 10)), (255, 0, 255), 1)
        cv2.circle(image, (int(left_eyelid.x), (int(left_eyelid.y)+10)), 1, (255, 0, 255), -1)
        cv2.putText(image, 'Right MRD: {:.2f} mm'.format(real_distance_eyelid_left), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
        #Right Bifocal Length
        right_pupil_eyelid = ((((right_pupil.x - right_eyelid.x)**2) + ((right_pupil.y - right_eyelid.y + 10)**2))**0.5)
        real_distance_eyelid_right = right_pupil_eyelid/4.3
    
        cv2.line(image,(int(right_pupil.x), int(right_pupil.y)), (int(right_eyelid.x), (int(right_eyelid.y)+10)), (128, 0, 128), 1)
        cv2.circle(image, (int(right_eyelid.x), (int(right_eyelid.y)+10)), 1, (128, 0, 128), -1)
        cv2.putText(image, 'Left MRD: {:.2f} mm'.format(real_distance_eyelid_right), (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
    
        # Total PD
        # drawing the lines to find the pupil
        for ((a, b), (c, d)) in [(shape[37], shape[40]), (shape[38], shape[41]), (shape[43], shape[46]), (shape[47], shape[44])]:
            cv2.line(image, (a, b), (c, d), (255, 0, 0), 1)

        # drawing central points and showing information
        cv2.circle(image, (int(left_pupil.x), int(left_pupil.y)), 1, (0, 0, 255), -1)
        cv2.circle(image, (int(right_pupil.x), int(right_pupil.y)), 1, (0, 0, 255), -1)

        cv2.line(image, (int(left_pupil.x), int(left_pupil.y)), (int(right_pupil.x), int(right_pupil.y)), (0, 255, 0), 1)
    
        x = left_pupil.x
        y = right_pupil.x

        pd = (y - x)/4.3
        print(pd)
        pupil_distance = Pupil.distance(left_pupil, right_pupil)

        cv2.putText(image, 'distance: {:.2f} mm'.format(pd), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
        # Nose bridge coordinate
        nose_bridge = shape[27]  # Nose bridge (between the eyes)
        cv2.circle(image, (nose_bridge[0], nose_bridge[1]), 2, (255, 0, 0), -1)

        # Left Pupil to Nose
        cv2.line(image, (int(left_pupil.x), int(left_pupil.y)), 
                    (nose_bridge[0], nose_bridge[1]), (255, 255, 0), 1)

        # Calculate left pupil to nose bridge distance
        distance_to_nose_bridge_LP = ((((left_pupil.x - nose_bridge[0])**2) + ((left_pupil.y - nose_bridge[1])**2))**0.5)
        real_distance_LP = distance_to_nose_bridge_LP/4.3
    
        cv2.putText(image, 'Right PD: {:.2f} mm'.format(real_distance_LP), 
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)    
    
        # Right Pupil to Nose
        cv2.line(image, (int(right_pupil.x), int(right_pupil.y)), 
                    (nose_bridge[0], nose_bridge[1]), (0, 0, 255), 1)

        # Calculate right pupil to nose bridge distance
        distance_to_nose_bridge_RP = ((((right_pupil.x - nose_bridge[0])**2) + ((right_pupil.y - nose_bridge[1])**2))**0.5)
        # real_distance_RP = distance_to_nose_bridge_RP/4.3
        real_distance_RP = pd - real_distance_LP
    
        cv2.putText(image, 'Left PD: {:.2f} mm'.format(real_distance_RP), 
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)       # BGR
        
        
        for result in results:
            boxes = result.boxes  # Get bounding boxes
            if content == 1:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list
                    confidence = box.conf[0].item()  # Get confidence score
                    class_id = int(box.cls[0].item())  # Get class ID

                    print(f"Class ID: {class_id}, Confidence: {confidence:.2f}")
                    print(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                    # Extract Bottom-Left Coordinate (x1, y2)
                    bottom_left_x = int(x1)
                    bottom_left_y = int(y2) - 8

                    # Draw a line from left pupil to bottom-left coordinate of specs
                    cv2.line(image, (int(left_pupil.x), int(left_pupil.y)), 
                             (int(left_pupil.x), int(bottom_left_y)), (0, 0, 255), 1)

                    # Calculate Euclidean distance
                    distance_to_specs_frame = ((((left_pupil.x - left_pupil.x)**2) + 
                                                ((left_pupil.y - bottom_left_y)**2))**0.5)
                    real_distance_sf = distance_to_specs_frame / 4.3  # Adjusted scale

                    # Display Distance on Image
                    cv2.putText(image, 'Bifocal for Specs: {:.2f} mm'.format(real_distance_sf), 
                                (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(image, 'Bifocal for Specs: No specs', (20, 300), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            

    # show the output image with the face detection + facial landmarks
    cv2.imshow('Output', image)
    cv2.waitKey(1)     

    cv2.imwrite('output/{}.jpg'.format(time.time()), image)


"""
Color Notations:

Distance = Green
Left PD = Red
Right PD = Blue
Left MRD = Pink
Right MRD = Yellow

"""