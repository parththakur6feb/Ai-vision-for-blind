import pyttsx3
import threading
from queue import Queue
from config.settings import VOICE
import platform
import time
import subprocess

_tts_queue: "Queue[str]" = Queue()
_tts_thread_started = False
_lock = threading.Lock()
_last_spoken_text = ""
_last_spoken_time = 0.0
_speak_cooldown_seconds = 1.0

def _tts_worker():
    engine = None
    while True:
        text = _tts_queue.get()
        try:
            if not text or not str(text).strip():
                continue

            system_name = platform.system()
            if system_name == "Windows":
                # Prefer Windows .NET System.Speech via PowerShell for reliability
                try:
                    safe_text = text.replace("'", "`'")
                    rate = int(VOICE.get("rate", 160))
                    volume = int(float(VOICE.get("volume", 1.0)) * 100)
                    ps_cmd = (
                        "Add-Type -AssemblyName System.Speech; "
                        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$s.Rate = {max(-10, min(10, (rate-160)//20))}; " ""
                        f"$s.Volume = {max(0, min(100, volume))}; "
                        f"$s.Speak('{safe_text}')"
                    )
                    subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True)
                    print("Speak done via System.Speech")
                    continue
                except Exception:
                    # Fallback to pyttsx3 if PowerShell path fails
                    driver_name = 'sapi5'
                    engine = pyttsx3.init(driverName=driver_name)
                    engine.setProperty('rate', VOICE.get("rate", 160))
                    engine.setProperty('volume', VOICE.get("volume", 1.0))
            else:
                # Lazily initialize once for non-Windows
                if engine is None:
                    driver_name = 'nsss' if system_name == 'Darwin' else None
                    engine = pyttsx3.init(driverName=driver_name)
                    engine.setProperty('rate', VOICE.get("rate", 160))
                    engine.setProperty('volume', VOICE.get("volume", 1.0))

            print(f"Speak: {text}")
            try:
                engine.stop()
            except Exception:
                pass
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                # Attempt one-time reinitialization and retry
                try:
                    system_name = platform.system()
                    driver_name = 'sapi5' if system_name == 'Windows' else ('nsss' if system_name == 'Darwin' else None)
                    engine = pyttsx3.init(driverName=driver_name)
                    engine.setProperty('rate', VOICE.get("rate", 160))
                    engine.setProperty('volume', VOICE.get("volume", 1.0))
                    if system_name == 'Windows':
                        try:
                            voices = engine.getProperty('voices')
                            preferred_voice_id = None
                            for v in voices:
                                name = getattr(v, 'name', '') or ''
                                id_ = getattr(v, 'id', '') or ''
                                if 'Zira' in name or 'Zira' in id_:
                                    preferred_voice_id = v.id
                                    break
                                if 'David' in name or 'David' in id_:
                                    preferred_voice_id = v.id
                            if preferred_voice_id:
                                engine.setProperty('voice', preferred_voice_id)
                        except Exception:
                            pass
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    # bubble up to fallback
                    raise
        except Exception as e:
            # Fallback to macOS 'say' if available
            try:
                if platform.system() == 'Darwin':
                    subprocess.run(['say', text])
            except Exception:
                pass
            try:
                print(f"TTS error: {e}")
            except Exception:
                pass
        finally:
            _tts_queue.task_done()

def _ensure_worker():
    global _tts_thread_started
    with _lock:
        if not _tts_thread_started:
            thread = threading.Thread(target=_tts_worker, daemon=True)
            thread.start()
            _tts_thread_started = True

def speak(text: str):
    _ensure_worker()
    global _last_spoken_text, _last_spoken_time
    now = time.time()
    # Skip if same text repeated too quickly
    if not text or not str(text).strip():
        return
    if text == _last_spoken_text and (now - _last_spoken_time) < _speak_cooldown_seconds:
        return
    _last_spoken_text = text
    _last_spoken_time = now
    _tts_queue.put(text)
