import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st

# Load different YOLO models (assumed to be available in the same directory)
model_options = {
    'YOLOv5': YOLO('yolov5xu.pt'),
    'YOLOv8': YOLO('yolov8x-seg.pt')
}

# Image manipulation techniques
manipulation_options = ["None", "Grayscale", "Edge Detection", "Blur"]

# Function to process image with chosen model
def process_image(image, model):
    resized_image = image.resize((640, 640))
    resized_image_np = np.array(resized_image)

    # Detect and segment with chosen model
    results = model.predict(resized_image_np)
    for r in results[0].boxes.data.tolist():
        xmin, ymin, xmax, ymax, confidence, class_id = r
        label = model.names[int(class_id)]
        cv2.rectangle(resized_image_np, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(resized_image_np, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert final image to PIL format
    return Image.fromarray(resized_image_np)

# Function to apply image manipulation
def apply_manipulation(image_np, technique):
    if technique == "Grayscale":
        return cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    elif technique == "Edge Detection":
        return cv2.Canny(image_np, 100, 200)
    elif technique == "Blur":
        return cv2.GaussianBlur(image_np, (15, 15), 0)
    return image_np

# Streamlit interface
st.title("Enhanced YOLO-based Image Manipulation Tool")
st.write("Choose a detection model and manipulation technique.")

# Model selection in Streamlit
model_choice = st.selectbox("Choose YOLO Model", list(model_options.keys()))
chosen_model = model_options[model_choice]

# Manipulation technique selection
manipulation_choice = st.selectbox("Choose Manipulation Technique", manipulation_options)

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)

    # Process and manipulate
    processed_image = process_image(image, chosen_model)
    processed_image_np = np.array(processed_image)
    manipulated_image = apply_manipulation(processed_image_np, manipulation_choice)

    # Display result
    st.image(manipulated_image, caption='Processed Image', use_column_width=True)
    st.success("Image processed successfully!")
