import speech_recognition as sr

# Function that will be called upon saying "activate"
def target_function():
    print("Activation successful! Target function is called.")

# Function that listens for the "activate" command
def listen_for_activation():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for the command...")
        audio = recognizer.listen(source)

        try:
            # Recognize speech using Google Web Speech API
            command = recognizer.recognize_google(audio).lower()
            print(f"Command received: {command}")

            # Check if the command is "activate"
            if "activate" in command:
                target_function()
            else:
                print("Command not recognized. Say 'activate' to trigger the function.")

        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
        except sr.RequestError:
            print("Could not request results; check your network connection.")

# Usage
listen_for_activation()
