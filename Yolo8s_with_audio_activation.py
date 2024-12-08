from ultralytics import YOLO
import cv2
import cvzone
import math
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import pygame

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8s")

# Object detection class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", 
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", 
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", 
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
              "scissors", "teddy bear", "hair drier", "toothbrush"]

# Function to convert text to speech
def text_to_audio(text, language='en'):
    tts = gTTS(text=text, lang=language, slow=False)
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)

    pygame.mixer.init()
    pygame.mixer.music.load(audio_data, 'mp3')
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

# Function to listen for voice commands
def listen_for_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for a command...")
        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            return command
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return None
        except sr.RequestError:
            print("Network error.")
            return None

# Object detection function
def object_detection():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                # Speak detected object
                text_to_audio(f"{currentClass} detected")
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)))

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break
        # Check for "Stop" command
        command = listen_for_command()
        if command and "stop" in command:
            text_to_audio("Stopping object detection")
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    while True:
        command = listen_for_command()
        if command:
            if "activate" in command:
                text_to_audio("Object detection activated.")
                object_detection()
            elif "exit" in command:
                text_to_audio("Exiting the program.")
                break
            else:
                print("Command not recognized. Say 'activate' or 'exit'.")

# Start the program
main()
