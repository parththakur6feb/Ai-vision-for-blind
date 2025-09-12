import os
import cv2
import numpy as np

# Try to import the dlib-based face_recognition library.
# If unavailable (e.g., dlib build issues), fall back to OpenCV-only detection.
try:
    import face_recognition  # type: ignore
    HAS_FACE_RECOGNITION = True
except Exception:
    face_recognition = None
    HAS_FACE_RECOGNITION = False

class FaceRecognizer:
    def __init__(self):
        # These are the attributes used in recognition
        self.known_face_encodings = []
        self.known_face_names = []
        if not HAS_FACE_RECOGNITION:
            # Prepare OpenCV Haar cascade for face detection as a fallback
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
            # Prepare LBPH recognizer to classify known faces by their cropped images
            try:
                self._lbph = cv2.face.LBPHFaceRecognizer_create()
            except Exception:
                self._lbph = None
            self._lbph_trained = False
            self._name_to_label = {}
            self._label_to_name = []
            self._lbph_images = []
            self._lbph_labels = []

    def load_known_faces(self, folder_path):
        """
        Load all images from the 'known_faces' folder
        and extract their face encodings.
        """
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(folder_path, filename)
                name = os.path.splitext(filename)[0]
                if HAS_FACE_RECOGNITION:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                else:
                    # For LBPH fallback, collect grayscale face regions and labels
                    if getattr(self, "_haar_cascade", None) is not None:
                        img = cv2.imread(image_path)
                        if img is None:
                            continue
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = self._haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                        for (x, y, w, h) in faces:
                            roi = gray[y:y+h, x:x+w]
                            roi_resized = cv2.resize(roi, (200, 200))
                            if name not in self._name_to_label:
                                self._name_to_label[name] = len(self._label_to_name)
                                self._label_to_name.append(name)
                            label_id = self._name_to_label[name]
                            self._lbph_images.append(roi_resized)
                            self._lbph_labels.append(label_id)

        # Train LBPH if available and data present
        if not HAS_FACE_RECOGNITION and self._lbph is not None and len(self._lbph_images) > 0:
            try:
                self._lbph.train(self._lbph_images, np.array(self._lbph_labels))
                self._lbph_trained = True
            except Exception:
                self._lbph_trained = False

    def recognize_faces(self, frame):
        """
        Detect faces in the frame and return a list of dictionaries:
        {'bbox': (x1, y1, x2, y2), 'name': name}
        """
        results = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if HAS_FACE_RECOGNITION:
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if best_match_index < len(matches) and matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                results.append({
                    "bbox": (left, top, right, bottom),
                    "name": name
                })
            return results

        # Fallback path: OpenCV Haar cascade detection only (no recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            left, top, right, bottom = x, y, x + w, y + h
            name = "Unknown"
            if getattr(self, "_lbph", None) is not None and self._lbph_trained and len(self._label_to_name) > 0:
                try:
                    roi = gray[y:y+h, x:x+w]
                    roi_resized = cv2.resize(roi, (200, 200))
                    label_id, confidence = self._lbph.predict(roi_resized)
                    # Lower confidence means a better match in LBPH (typical threshold around 80)
                    if confidence < 85 and 0 <= label_id < len(self._label_to_name):
                        name = self._label_to_name[label_id]
                except Exception:
                    pass
            results.append({
                "bbox": (left, top, right, bottom),
                "name": name
            })
        return results
