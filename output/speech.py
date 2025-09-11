import pyttsx3
from config.settings import VOICE

engine = pyttsx3.init()
engine.setProperty('rate', VOICE["rate"])
engine.setProperty('volume', VOICE["volume"])

def speak(text):
    engine.say(text)
    engine.runAndWait()
