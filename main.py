
from input.camera import Camera
from input.microphone import VoiceCommand
from processing.object_detection import ObjectDetector
from processing.ocr import OCRReader
from processing.face_recognition import FaceRecognizer
from output.speech import speak
import cv2
import pytesseract
import os
import threading
from queue import Queue
import time
import platform
from collections import deque
from typing import List, Tuple, Dict

# Point pytesseract to your Tesseract installation (OS-aware)
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
else:
    possible_paths = [
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break

# Initialize modules
cam = Camera()
mic = VoiceCommand()
detector = ObjectDetector()
ocr = OCRReader()
face_recog = FaceRecognizer()

# Load known faces
face_folder = os.path.join(os.getcwd(), "known_faces")
face_recog.load_known_faces(face_folder)

# Command queue
commands = Queue()
running = True

# Recent announcements to avoid spamming speech
recent_objects = deque(maxlen=5)
last_object_announce = 0.0
OBJECT_DEBOUNCE_SECONDS = 2.0
DETECTION_INTERVAL_SECONDS = 0.25  # run heavy detection at ~4 FPS
last_detection_time = 0.0
# Visual marker controls
MARKER_DEBOUNCE_SECONDS = 1.5
MAX_ANNOTATIONS_ON_SCREEN = 60
last_label_marker_time: Dict[str, float] = {}

# Continuous visuals/speech toggles (set both False to show markers only on commands)
ENABLE_CONTINUOUS_MARKERS = False
ENABLE_CONTINUOUS_OBJECT_SPEECH = False

# Annotations buffer drawn in the main loop
annotations: List[Dict] = []
annotations_lock = threading.Lock()
ANNOTATION_TTL_SECONDS = 2.0

def add_annotation(kind: str, bbox: Tuple[int, int, int, int], caption: str, color: Tuple[int, int, int]):
    with annotations_lock:
        annotations.append({
            "t": time.time(),
            "kind": kind,
            "bbox": bbox,
            "caption": caption,
            "color": color
        })

def draw_annotations(frame):
    now = time.time()
    with annotations_lock:
        # filter expired and draw active
        active = []
        for ann in annotations:
            if (now - ann["t"]) <= ANNOTATION_TTL_SECONDS:
                active.append(ann)
                x1, y1, x2, y2 = ann["bbox"]
                color = ann["color"]
                caption = ann["caption"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, caption, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.circle(frame, (x1, y1), 4, color, -1)
        # replace with active list only
        if len(active) > MAX_ANNOTATIONS_ON_SCREEN:
            active = active[-MAX_ANNOTATIONS_ON_SCREEN:]
        annotations.clear()
        annotations.extend(active)

# Thread to continuously listen to commands
def listen_thread():
    global running
    while running:
        cmd = mic.listen()
        if cmd:
            print(f"Heard command: {cmd}")
            commands.put(cmd.lower())

# Start listening thread
threading.Thread(target=listen_thread, daemon=True).start()

def process_command(command, frame):
    if command.strip() == "object":
        objects = detector.detect(frame)
        for obj in objects:
            label = obj.get("label", "Unknown")
            bbox = obj.get("bbox", None)
            direction = obj.get("direction", "")
            speak(f"Object {label} ahead {direction}")
            if bbox:
                x1, y1, x2, y2 = bbox
                add_annotation("object", (x1, y1, x2, y2), f"Object: {label}", (0, 200, 0))

    elif command.strip() == "read":
        text = ocr.read(frame)
        speak(text)
        data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            add_annotation("text", (x, y, x + w, y + h), "Text", (255, 0, 0))

    elif command.strip() == "who":
        faces = face_recog.recognize_faces(frame)
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            name = face.get('name', 'Unknown') or 'Unknown'
            add_annotation("person", (x1, y1, x2, y2), f"Person: {name}", (0, 0, 200))
            speak(f"Person {name}")

    elif command.strip() == "exit":
        global running
        running = False
        speak("Exiting system...")
        try:
            cam.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        return

def main():
    global last_detection_time, running
    speak("System ready. Say a command ('object', 'read', 'who', 'exit').")

    while running:
        try:
            if not running:
                break
            frame = cam.get_frame()
            if frame is None:
                if not running:
                    break
                continue

            # Optional continuous detection/markers and object speech (disabled by default)
            if ENABLE_CONTINUOUS_MARKERS or ENABLE_CONTINUOUS_OBJECT_SPEECH:
                try:
                    now = time.time()
                    if (now - last_detection_time) >= DETECTION_INTERVAL_SECONDS:
                        objects = detector.detect(frame)
                        # cache detection time
                        last_detection_time = now
                        if ENABLE_CONTINUOUS_MARKERS:
                            # render markers (debounced per label)
                            for obj in objects:
                                label = obj.get("label", "Unknown")
                                x1, y1, x2, y2 = obj.get("bbox", (0, 0, 0, 0))
                                last_t = last_label_marker_time.get(label, 0.0)
                                if (now - last_t) >= MARKER_DEBOUNCE_SECONDS:
                                    add_annotation("object", (x1, y1, x2, y2), f"Object: {label}", (0, 200, 0))
                                    last_label_marker_time[label] = now
                        if ENABLE_CONTINUOUS_OBJECT_SPEECH:
                            # Debounced TTS for objects
                            global last_object_announce
                            if objects and (now - last_object_announce) > OBJECT_DEBOUNCE_SECONDS:
                                speak(", ".join(sorted({obj.get('label','Unknown') for obj in objects})))
                                last_object_announce = now
                except Exception:
                    pass

            # Process command if any (for OCR and faces and explicit queries)
            if not commands.empty():
                command = commands.get()
                threading.Thread(target=process_command, args=(command, frame.copy()), daemon=True).start()

            # Always show live feed with markers
            draw_annotations(frame)
            cv2.imshow("Drishiti Setu - Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                speak("Exiting system...")
                try:
                    cam.release()
                except Exception:
                    pass
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
                break

        except RuntimeError:
            if not running:
                break
            print("Failed to grab frame")
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
