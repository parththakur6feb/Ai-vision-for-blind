import cv2

class Camera:
    def __init__(self, device_index=0):
        self.cap = None
        backend = cv2.CAP_DSHOW  # Prefer DirectShow on Windows
        candidate_indices = [device_index] + [i for i in range(4) if i != device_index]
        for idx in candidate_indices:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            # Attempt to read one frame to verify
            ret, _ = cap.read()
            if ret:
                self.cap = cap
                break
            cap.release()

        if self.cap is None:
            raise RuntimeError("Failed to open camera on indices 0-3. Check camera permissions and device index.")
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to grab frame")
        return frame
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
