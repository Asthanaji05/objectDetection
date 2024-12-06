import torch
from torchvision import models, transforms
import cv2  # OpenCV for video capture
import numpy as np

# COCO dataset class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load a pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define a function to process a single frame
def process_frame(frame, threshold=0.5):
    # Transform frame to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    frame_tensor = transform(frame).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(frame_tensor)
    
    # Extract detections from output
    detections = outputs[0]
    boxes = detections["boxes"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()

    # Draw detections on the frame
    for i in range(len(boxes)):
        if scores[i] >= threshold:
            # Get box coordinates
            x_min, y_min, x_max, y_max = boxes[i]
            # Ensure label is within valid range
            label_idx = labels[i]
            if 0 < label_idx < len(COCO_CLASSES):
                class_name = COCO_CLASSES[label_idx]
            else:
                class_name = "Unknown"
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            # Add label and score
            label = f"{class_name}: {scores[i]:.2f}"
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Apply Canny edge detection
def apply_canny(frame, low_threshold=50, high_threshold=150):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    # Convert edges back to BGR format for visualization
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_rgb

# Main code for video capture
if __name__ == "__main__":
    # Open the webcam (use 0 for the default camera)
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video capture.")
        exit()

    print("Press 'q' to quit the real-time detection.")
    while True:
        # Capture a frame from the video feed
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame from BGR (OpenCV format) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Faster R-CNN
        processed_frame = process_frame(frame_rgb)

        # Apply Canny edge detection on the frame
        edges_frame = apply_canny(frame)

        # Stack the edge-detected frame and the processed frame side by side
        combined_frame = np.hstack((edges_frame, processed_frame))

        # Display the combined frame
        cv2.imshow("Real-Time Object Detection with Canny Edge Detection", combined_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()
