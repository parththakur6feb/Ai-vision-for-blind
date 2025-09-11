import face_recognition
import os
import cv2
import numpy as np

class FaceRecognizer:
    def __init__(self):
        # These are the attributes used in recognition
        self.known_face_encodings = []
        self.known_face_names = []

    def load_known_faces(self, folder_path):
        """
        Load all images from the 'known_faces' folder
        and extract their face encodings.
        """
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(folder_path, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(os.path.splitext(filename)[0])

    def recognize_faces(self, frame):
        """
        Detect faces in the frame and return a list of dictionaries:
        {'bbox': (x1, y1, x2, y2), 'name': name}
        """
        results = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            results.append({
                "bbox": (left, top, right, bottom),
                "name": name
            })
        return results
