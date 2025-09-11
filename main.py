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

# Point pytesseract to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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

# Thread to continuously listen to commands
def listen_thread():
    while True:
        cmd = mic.listen()
        if cmd:
            commands.put(cmd.lower())

# Start listening thread
threading.Thread(target=listen_thread, daemon=True).start()

def process_command(command, frame):
    if "object" in command:
        objects = detector.detect(frame)
        for obj in objects:
            label = obj.get("label", "Unknown")
            bbox = obj.get("bbox", None)
            direction = obj.get("direction", "")
            speak(f"{label} ahead {direction}")
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    elif "read" in command:
        text = ocr.read(frame)
        speak(text)
        data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    elif "who" in command:
        faces = face_recog.recognize_faces(frame)
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            name = face['name']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            speak(f"{name}")

    elif "exit" in command:
        speak("Exiting system...")
        cam.release()
        cv2.destroyAllWindows()
        os._exit(0)  # Exit immediately

def main():
    speak("System ready. Say a command ('object', 'read', 'who', 'exit').")

    while True:
        try:
            frame = cam.get_frame()
            if frame is None:
                continue

            # Process command if any
            if not commands.empty():
                command = commands.get()
                threading.Thread(target=process_command, args=(command, frame.copy()), daemon=True).start()

            # Always show live feed with markers
            cv2.imshow("Drishiti Setu - Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                speak("Exiting system...")
                cam.release()
                cv2.destroyAllWindows()
                break

        except RuntimeError:
            print("Failed to grab frame")
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
