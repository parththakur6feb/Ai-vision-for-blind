# AI Vision for Blind

**AI Vision for Blind** is a Python-based assistive application designed to help visually impaired users interact with their environment by recognizing objects, people, and providing audio feedback in real time.

---

## 🚀 Features

- Recognizes known faces using a stored face database  
- Detects objects and common obstacles  
- Converts input to audio descriptions  
- Continuous input (camera) and offline-first model usage where possible  

---

## 🧰 Tech Stack & Files

| Component | Description |
|-----------|-------------|
| `main.py` | Entry point — orchestrates vision, detection, feedback loops |
| `testcam.py` | Testing module for camera feed and image capture |
| `known_faces/` | Directory to store and manage images of known persons |
| `models/` | Pre-trained ML models used for object & face detection |
| `utils/` | Utility helpers (image processing, audio conversion, etc.) |

---

## 🔧 Setup

1. Clone the repo:  
   ```bash
   git clone https://github.com/parththakur6feb/Ai-vision-for-blind.git
   cd Ai-vision-for-blind
