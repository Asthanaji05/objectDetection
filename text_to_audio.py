from gtts import gTTS
from io import BytesIO
import pygame

def text_to_audio(text, language='en'):
    # Generate audio data
    tts = gTTS(text=text, lang=language, slow=False)
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)

    # Initialize and play audio using pygame
    pygame.mixer.init()
    pygame.mixer.music.load(audio_data, 'mp3')
    pygame.mixer.music.play()
    
    # Wait until the audio finishes playing
    while pygame.mixer.music.get_busy():
        continue

# Usage
text_to_audio("Hello, this audio is played directly.")
