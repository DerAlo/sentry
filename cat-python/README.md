# Cat Tracking System - Python Port mit CUDA Support

## 🚀 **WARUM PYTHON?**
- **Echte CUDA-Unterstützung** mit PyTorch/TensorRT
- **Ultraperformante YOLO-Inferenz** mit GPU-Beschleunigung
- **Bessere Computer Vision Bibliotheken** (OpenCV, PyTorch, Ultralytics)
- **Native GPU-Integration** ohne Java-Limitierungen

## 📁 **Projekt-Struktur**

```
cat-python/
├── server/                 # Raspberry Pi Server
│   ├── camera_server.py   # Webcam Streaming Server
│   ├── serial_controller.py # Arduino Serial Communication
│   └── requirements.txt   # Server Dependencies
├── client/                 # PC Client mit CUDA
│   ├── yolo_tracker.py    # CUDA-beschleunigtes YOLO
│   ├── gui_client.py      # JavaFX-ähnliche GUI
│   ├── network_client.py  # Server Communication
│   └── requirements.txt   # Client Dependencies (CUDA)
├── shared/                 # Gemeinsame Module
│   ├── protocol.py        # Communication Protocol
│   └── config.py          # Configuration Classes
└── models/                 # YOLO Models
    └── yolov9c.pt         # YOLOv9 PyTorch Model
```

## 🎯 **Features (1:1 Java Port)**

### **Server (Raspberry Pi)**
- ✅ **Webcam HTTP Streaming** (Flask/FastAPI)
- ✅ **WebSocket Servo Control** 
- ✅ **Arduino Serial Communication**
- ✅ **Gleiche API** wie Java-Version

### **Client (PC mit CUDA)**
- ✅ **CUDA-natives YOLO** (PyTorch + TensorRT)
- ✅ **Echtzeit Object Tracking**
- ✅ **GUI mit Tkinter/PyQt**
- ✅ **Servo Control Integration**
- ✅ **Friend/Enemy Classification**

## ⚡ **CUDA Performance Erwartung**
- **Java CPU:** ~5-10 FPS
- **Python CUDA:** **30-60+ FPS** 🚀
- **TensorRT optimiert:** **100+ FPS** möglich!

## 🛠 **Installation**

### **Client (PC):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics tensorrt opencv-python-headless
```

### **Server (Raspberry Pi):**
```bash
pip install flask websockets opencv-python pyserial
```

## 🚀 **Quick Start**

### **1. Server starten (Raspberry Pi):**
```bash
cd server
python camera_server.py
```

### **2. Client starten (PC):**
```bash
cd client
python gui_client.py
```

## 🎯 **Das wird DEUTLICH schneller als Java!**
