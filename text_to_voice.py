# Import the required modules
from gtts import gTTS
import os
import platform
import subprocess

# The text that you want to convert to audio in Gujarati
mytext = 'તમે ગીક્સફોર્ગીક્સ પર સ્વાગત છે જો!'

# Language in which you want to convert (Gujarati)
language = 'gu'

# Passing the text and language to the engine
try:
    myobj = gTTS(text=mytext, lang=language, slow=False)

    # Define the output path for the mp3 file
    output_path = r"C:\Users\hp\Desktop\welcome_gujarati.mp3"

    # Saving the converted audio to the file
    myobj.save(output_path)
    print(f"Audio file saved at {output_path}")

except Exception as e:
    print(f"Error in text-to-speech conversion: {e}")

# Playing the converted file
try:
    # Check the operating system to use the correct command
    if platform.system() == "Windows":
        subprocess.run(['start', output_path], shell=True)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(['open', output_path])
    else:  # Linux/Unix
        subprocess.run(['xdg-open', output_path])

    print("Playing the audio file...")

except Exception as e:
    print(f"Error playing the audio file: {e}")
