import streamlit as st
import cv2
import cvzone
import math
import time
from ultralytics import YOLO
import numpy as np

# Initialize YOLO model (nano version for faster inference)
model = YOLO("../Yolo-Weights/yolov8n")

# Class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", 
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", 
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", 
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
              "scissors", "teddy bear", "hair drier", "toothbrush"]

# Function to process video or image
def process_input(input_type="camera"):
    # Start timer
    start_time = time.time()
    
    if input_type == "camera":
        # Initialize camera with reduced resolution
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # Width
        cap.set(4, 480)  # Height
        stframe = st.empty()  # Create a placeholder for the video frame

        while True:
            success, img = cap.read()

            # Run YOLO inference with streaming for efficiency
            results = model(img, stream=True)

            for r in results:
                for box in r.boxes:
                    conf = box.conf[0]

                    # Skip low-confidence detections
                    if conf < 0.5:
                        continue

                    # Extract bounding box coordinates and class information
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the bounding box
                    radius = max(x2 - x1, y2 - y1) // 2  # Size of the circle (radius based on object size)
                    currentClass = classNames[int(box.cls[0])]

                    # Draw circle (heatmap-style) at the center of the detected object
                    if draw_option == "Circle":
                        cv2.circle(img, (cx, cy), radius, (0, 255, 0), 2)  # Green circle
                    else:
                        # Draw bounding box for "Square"
                        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1))

                    # Display text for object class and confidence
                    cvzone.putTextRect(img, f'{currentClass} {conf:.2f}', (x1, max(35, y1)))

            # Convert the frame to RGB and display on Streamlit
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            stframe.image(img_rgb, channels="RGB", use_container_width=True)  # Updated parameter

            # Allow to stop the video stream by pressing 'q'
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()

    elif input_type == "image":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            # Read the uploaded image
            img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

            # Run YOLO inference
            results = model(img, stream=True)

            for r in results:
                for box in r.boxes:
                    conf = box.conf[0]

                    # Skip low-confidence detections
                    if conf < 0.5:
                        continue

                    # Extract bounding box coordinates and class information
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the bounding box
                    radius = max(x2 - x1, y2 - y1) // 2  # Size of the circle (radius based on object size)
                    currentClass = classNames[int(box.cls[0])]

                    # Draw circle (heatmap-style) at the center of the detected object
                    if draw_option == "Circle":
                        cv2.circle(img, (cx, cy), radius, (0, 255, 0), 2)  # Green circle
                    else:
                        # Draw bounding box for "Square"
                        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1))

                    # Display text for object class and confidence
                    cvzone.putTextRect(img, f'{currentClass} {conf:.2f}', (x1, max(35, y1)))

            # Convert the frame to RGB for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, channels="RGB", use_container_width=True)

    # End timer and show processing time
    end_time = time.time()
    st.write(f"Processing time: {end_time - start_time:.2f} seconds")

# Streamlit UI Setup
st.title("YOLO Object Detection with Heatmap Circles vs Bounding Boxes")
st.write("This app uses the YOLO object detection model to detect objects and show results using circles (heatmap) or bounding boxes.")

# Option to choose input method
input_option = st.radio("Select input method", ("Camera", "Upload Image"))

# Option to choose visualization method
draw_option = st.selectbox("Select visualization method", ("Square", "Circle"))

if input_option == "Camera":
    st.button("Start Detection", on_click=process_input, args=("camera",))
else:
    process_input("image")
