# Object Detection
## Models:
- Yolo8n : Shashank Asthana's Model
- Yolo5_8 : Dev Bansal's Model
- LiveRCNN: Shreya Asthana's Model
- Yolov4 : Shreya Tripathi's Model

## Helping Code 
- text_to_audio (for output)
- activate (input) 

## My Model : Yolo8s_with_audio_activation
Integrated Yolov8 , NLP, Voice Commands 

## System Architecture:

---

1. **Camera**  
   - Captures real-time images or video frames from the environment.  
   - Sends the image stream to the next stage.  

2. **Object Detection Module**  
   - Processes the images using the YOLO model.  
   - Detects and classifies objects within the frame.  
   - Outputs object names, bounding boxes, and confidence scores.  
   - Estimates whether objects are "closer" or "farther" based on the size of their bounding boxes.

3. **Voice Command Module**  
   - **Microphone Input**: Listens to user commands using a speech recognition library (e.g., SpeechRecognition).  
   - Processes commands like "activate," "stop," or "exit."  

4. **Text-to-Speech Module**  
   - Converts detected objects, their proximity (e.g., "closer" or "farther"), and scene descriptions into spoken feedback using Google Text-to-Speech (gTTS).  
   - Plays the audio output through the speaker.

5. **AI Response System**  
   - Combines object detection results, including proximity estimation (closer/farther), with voice input.  
   - Provides intelligent responses to user commands and describes the detected scene (e.g., "The scene contains a person and a bicycle, both are closer").  

6. **User Interaction**  
   - Outputs include a live video stream with annotations (bounding boxes and labels), and audible scene descriptions indicating the proximity of objects.  
   - Users interact through voice commands to control the system.

---

**Flow of Data:**  
Camera → Object Detection Module (with proximity estimation) → AI Response System → (Text-to-Speech and Display) → User Interaction
