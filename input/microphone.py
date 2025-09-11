import vosk, pyaudio, json

class VoiceCommand:
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.mic = pyaudio.PyAudio()
        self.stream = self.mic.open(format=pyaudio.paInt16, channels=1,
                                    rate=16000, input=True, frames_per_buffer=8192)
        self.stream.start_stream()

    def listen(self):
        data = self.stream.read(4096, exception_on_overflow=False)
        if self.recognizer.AcceptWaveform(data):
            result = json.loads(self.recognizer.Result())
            return result.get("text", "")
        return ""
