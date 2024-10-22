import speech_recognition as sr
import pyttsx3
from PIL import Image
import os

r = sr.Recognizer()


def SpeakText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

word_to_image = {
    "સફરજન": r"C:\Users\hp\Downloads\speechToText\speechToText\images\apple.jpg",  # Apple
    "કેળું": r"C:\Users\hp\Downloads\speechToText\speechToText\images\banana.jpeg",
    "કેળુ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\banana.jpeg",# Banana
    "બિલાડી":r"C:\Users\qhp\Downloads\speechToText\speechToText\images\cat.jpeg",    # Cat
    "કૂતરો": r"C:\Users\hp\Downloads\speechToText\speechToText\images\dog.jpeg",
    "કુતરો": r"C:\Users\hp\Downloads\speechToText\speechToText\images\dog.jpeg", # Dog
    # Gujarati Numbers
    "એક": r"C:\Users\hp\Downloads\speechToText\speechToText\images\one.jpg",
    "બે":  r"C:\Users\hp\Downloads\speechToText\speechToText\images\two.jpg",
    "ત્રણ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\three.jpg",
    "ચાર": r"C:\Users\hp\Downloads\speechToText\speechToText\images\four.jpg",
    "પાંચ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\five.jpg",
    "પાચ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\five.jpg",
    "છ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\six.jpg",
    "સાત": r"C:\Users\hp\Downloads\speechToText\speechToText\images\seven.jpg",
    "આઠ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\eight.jpg",
    "નૌ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\nine.jpg",
    "નવ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\nine.jpg",
    "નાઉ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\nine.jpg",
    "શૂન્ય": r"C:\Users\hp\Downloads\speechToText\speechToText\images\zero.jpg",
    "શુન્ય": r"C:\Users\hp\Downloads\speechToText\speechToText\images\zero.jpg",
    "શુંય": r"C:\Users\hp\Downloads\speechToText\speechToText\images\zero.jpg",
    "ક": r"C:\Users\hp\Downloads\speechToText\speechToText\images\ka.jpg",
    "ખ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\kha.jpg",
    "ગ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\ga.jpg",
    "ઘ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\gha.jpg",
    "ચ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\cha.jpg",
    "ભાઈ" : r"C:\Users\hp\Downloads\speechToText\speechToText\images\brother.jpeg",  # Bhai
    "બહેન": r"C:\Users\hp\Downloads\speechToText\speechToText\images\sister.jpeg",  # Bahen
    "બેન" : r"C:\Users\hp\Downloads\speechToText\speechToText\images\sister.jpeg",
    "યુવન": r"C:\Users\hp\Downloads\speechToText\speechToText\images\young.jpeg",
    "યુવાન": r"C:\Users\hp\Downloads\speechToText\speechToText\images\young.jpeg",
    "યૌવન": r"C:\Users\hp\Downloads\speechToText\speechToText\images\young.jpeg",  # Yauvan
    "પ્રેમ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\love.jpeg",    # Prem
    "તે":r"C:\Users\hp\Downloads\speechToText\speechToText\images\her.jpeg",  # Te
    "તેન": r"C:\Users\hp\Downloads\speechToText\speechToText\images\her.jpeg",
    "તરા":r"C:\Users\hp\Downloads\speechToText\speechToText\images\her.jpeg",
    "કૃપા": r"C:\Users\hp\Downloads\speechToText\speechToText\images\please.jpeg",  # Krupa
    "કરુપા":r"C:\Users\hp\Downloads\speechToText\speechToText\images\please.jpeg",
    "ક્રુપ્ય":r"C:\Users\hp\Downloads\speechToText\speechToText\images\please.jpeg",
    "કૃપ્ય":r"C:\Users\hp\Downloads\speechToText\speechToText\images\please.jpeg",
    "મદદ": r"C:\Users\hp\Downloads\speechToText\speechToText\images\help.jpeg",    # Madad
    "હું": r"C:\Users\hp\Downloads\speechToText\speechToText\images\me.jpeg", 
    "મળવા": r"C:\Users\hp\Downloads\speechToText\speechToText\images\find.jpeg",
    "માડવા": r"C:\Users\hp\Downloads\speechToText\speechToText\images\find.jpeg",
    "પુસ્તક": r"C:\Users\hp\Downloads\speechToText\speechToText\images\book.jpeg"   # Pustak
}


def display_image(word):
    if word in word_to_image:
        image_path = word_to_image[word]
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img.show()
        else:
            print(f"Image for {word} not found at {image_path}")
    else:
        print(f"No image mapped for word: {word}")


def log_recognized_text(text):
    with open("recognized_text.txt", "a", encoding="utf-8") as file:
        file.write(text + "\n")


while True:
    try:

        with sr.Microphone() as source2:
            print("Adjusting for ambient noise... Please wait.")
            r.adjust_for_ambient_noise(source2, duration=1)

            print("Listening...")
            audio2 = r.listen(source2)

            print("Recognizing speech...")
            MyText = r.recognize_google(audio2, language='gu-IN')
            MyText = MyText.lower()

            log_recognized_text(MyText)

            print(f"Recognized Text (logged to file): {MyText.encode('utf-8')}")

            SpeakText(MyText)

            words = MyText.split()

            for word in words:
                display_image(word)

    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio. Please try again.")

    except Exception as e:
        print(f"An error occurred: {e}")