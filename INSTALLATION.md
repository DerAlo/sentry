# 🐱 Cat Tracking System - Python CUDA Edition
## Hochperformante GPU-beschleunigte Objekterkennung

## 🚀 Schnellstart

### 1. Server Setup (Raspberry Pi)
```bash
cd server
pip install -r requirements.txt
python camera_server.py
```

### 2. Client Setup (PC mit CUDA GPU)
```bash
cd client
pip install -r requirements.txt
python gui_client.py
```

## 📋 Systemvoraussetzungen

### Server (Raspberry Pi):
- Raspberry Pi 4 (4GB+ RAM empfohlen)
- USB Webcam oder Pi Camera
- Arduino für Servo-Steuerung
- Python 3.8+

### Client (High-Performance PC):
- **NVIDIA GPU mit CUDA support** (GTX 1060+ / RTX series)
- CUDA Toolkit 11.8+ installiert
- Python 3.8+
- 8GB+ RAM empfohlen

## 🔧 Detaillierte Installation

### A) CUDA Setup (Client PC - KRITISCH!)

1. **NVIDIA Driver installieren:**
   - Neueste GeForce/Studio Treiber von nvidia.com
   - Mindestens Version 522.06+

2. **CUDA Toolkit installieren:**
   ```bash
   # Download von https://developer.nvidia.com/cuda-downloads
   # Version 11.8 oder 12.x empfohlen
   nvcc --version  # Überprüfen
   ```

3. **PyTorch mit CUDA installieren:**
   ```bash
   # Für CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # Für CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **CUDA-Test:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   print(f"GPU name: {torch.cuda.get_device_name(0)}")
   ```

### B) Server Installation (Raspberry Pi)

1. **Abhängigkeiten installieren:**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv
   sudo apt install libopencv-dev python3-opencv
   ```

2. **Virtual Environment:**
   ```bash
   python3 -m venv cat-env
   source cat-env/bin/activate
   pip install --upgrade pip
   ```

3. **Python Packages:**
   ```bash
   cd server
   pip install -r requirements.txt
   ```

4. **Arduino Setup:**
   - Arduino mit Servo-Steuerung anschließen
   - USB-Verbindung zum Raspberry Pi
   - Port in `server.properties` anpassen

### C) Client Installation (PC)

1. **Virtual Environment (empfohlen):**
   ```powershell
   python -m venv cat-cuda-env
   cat-cuda-env\Scripts\activate
   python -m pip install --upgrade pip
   ```

2. **CUDA PyTorch zuerst:**
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Weitere Dependencies:**
   ```powershell
   cd client
   pip install -r requirements.txt
   ```

4. **Ultralytics YOLO:**
   ```powershell
   pip install ultralytics
   # Auto-download von YOLOv8/v9 models beim ersten Start
   ```

## ⚙️ Konfiguration

### Server Config (`server/server.properties`):
```properties
# Webcam
camera.device=0
camera.width=640
camera.height=480
camera.fps=30

# Servo Control
arduino.port=/dev/ttyUSB0  # Linux
arduino.baudrate=9600

# Network
server.host=0.0.0.0
server.port=8080
websocket.port=8765
```

### Client Config:
```python
# In gui_client.py anpassen:
SERVER_HOST = "192.168.1.100"  # Raspberry Pi IP
SERVER_PORT = 8080
WEBSOCKET_PORT = 8765

# YOLO Model Selection
YOLO_MODEL = "yolov8n.pt"  # Schnell
# YOLO_MODEL = "yolov8s.pt"  # Ausgewogen  
# YOLO_MODEL = "yolov8m.pt"  # Hoch-Performance
```

## 🎯 Performance-Optimierung

### TensorRT (ULTRA Performance):
```python
# Automatische TensorRT-Optimierung beim ersten Start
# Erwartet: 100+ FPS bei RTX 4070+
model.export(format='engine')  # TensorRT compilation
```

### Multi-GPU Setup:
```python
# Für mehrere GPUs
device = torch.device('cuda:0')  # GPU 0
# device = torch.device('cuda:1')  # GPU 1
```

### Memory Optimization:
```python
# Für 4GB VRAM GPUs
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
```

## 🔍 Troubleshooting

### CUDA Probleme:
```bash
# Version check
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# Neuinstallation bei Problemen
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Performance Issues:
- **< 10 FPS:** Wahrscheinlich CPU-only, CUDA prüfen
- **10-30 FPS:** GPU läuft, aber nicht optimal
- **30-60 FPS:** Perfekt für real-time tracking
- **60+ FPS:** TensorRT läuft optimal

### Connection Issues:
```bash
# Server erreichbar?
ping 192.168.1.100

# Ports offen?
telnet 192.168.1.100 8080
telnet 192.168.1.100 8765
```

## 📊 Erwartete Performance

| Hardware | FPS (YOLO) | Latenz | Bemerkung |
|----------|------------|---------|-----------|
| RTX 4090 | 120+ FPS | <5ms | Ultra Performance |
| RTX 4070 | 80-100 FPS | <8ms | High Performance |
| RTX 3080 | 60-80 FPS | <10ms | Sehr gut |
| RTX 3060 | 40-60 FPS | <15ms | Gut |
| GTX 1660 | 25-35 FPS | <20ms | Akzeptabel |
| CPU Only | 5-10 FPS | >100ms | Langsam (wie Java) |

## 🎮 Verwendung

### 💻 **Lokaler Kamera-Modus**

```powershell
cd client
python gui_client.py
```

**Lokale Kamera aktivieren:**
   - Radio Button "Local Camera" wählen
   - "🔍 Scan Cameras" klicken
   - Gewünschte Kamera aus Dropdown wählen
   - "🎥 Start Camera" klicken

**Features im lokalen Modus:**
   - **CUDA YOLO Erkennung** (30-60 FPS)
   - **Freund-Feind-Klassifikation** mit farbigen Overlays
   - **Auto-Targeting** von Feinden
   - **Servo-Steuerung** via UI-Sliders
   - **Echtzeit-Tracking** ohne Server

### 🌐 **Server-Client Modus**

1. **Server starten:**
   ```bash
   cd server
   python camera_server.py
   ```

2. **Client starten:**
   ```powershell
   cd client
   python gui_client.py
   ```

**Verbindung herstellen:**
   - Radio Button "Network" wählen  
   - "🔌 Connect to Server" klicken
   - "🎯 Start Tracking" aktivieren
   - **Echte Servo-Steuerung** via Netzwerk
### 🎯 **Combat Features:**
- **👥 Freund-Feind-Verwaltung:** Add/Remove Buttons für dynamische Klassifizierung
- **🎯 Standard-Konfiguration:** Mensch = Freund, Katze = Feind
- **⚔️ Intelligente Logik:** Automatisches Hold-Fire bei Freunden im Bild
- **📹 Video-Aufzeichnung:** Automatisch bei Feindkontakt (optional)
- **🔄 Servo-Reset:** One-Click Zentrierung der Servos
- **🎮 Combat-Checkbox:** "SCHIESSBEFEHL" für Auto-Fire Aktivierung

### 🎯 **Tracking Features:**
- **Freunde (Grün):** Person, Dog (harmlos) - Hold Fire!
- **Feinde (Rot):** Cat, Bird (Auto-Targeting aktiv!)
- **Unbekannt (Gelb):** Andere Objekte
- **Auto-Targeting:** Automatisches Verfolgen von Feinden
- **Servo-Steuerung:** Real oder simuliert je nach Modus

## 🎯 Features

### 🎥 **Dual-Mode Operation (NEU!)**
- **Local Camera Mode:** Standalone Betrieb ohne Server
  - Direkte USB/Webcam-Nutzung
  - Kamera-Scanner für automatische Erkennung
  - Servo-Simulation via UI-Sliders
  - Vollständige CUDA-Beschleunigung
- **Network Mode:** Server-Client Architektur
  - Raspberry Pi Server mit Kamera
  - PC Client für CUDA-Processing
  - Echte Servo-Steuerung via WebSocket

### 🤖 **Auto-Targeting System:**
- **Freunde (Grün):** Monitor, Laptop (harmlos)
- **Feinde (Rot):** Person (Katzen-Störer) - **Automatisches Tracking!**
- **Unbekannt (Gelb):** Andere Objekte
- **Smart Targeting:** Höchste Konfidenz = Prioritätsziel
- **Servo Commands:** Automatische Nachführung des Ziels

### 🎮 **Manual Control:**
- **Servo-Steuerung:** Manuelle Kontrolle per Slider
- **Target-Auswahl:** Auswahl aus Erkennungsliste
- **Fire-Command:** Simulierter "Abschuss"-Button
- **Overlay-Toggle:** YOLO-Overlays ein/ausschalten

### ⚡ **Ultra Performance:**
- **Real-time CUDA YOLO** (30-60 FPS)
- **TensorRT Optimization** (100+ FPS möglich)
- **Multi-threaded Processing**
- **Memory-efficient Streaming**
- **GPU Memory Management**

## 🔄 Entwicklung

Das Python CUDA System ist die Zukunft des Cat Tracking Systems:
- **6x-10x schnellere** Objekterkennung durch echte GPU-Beschleunigung
- **Moderne Architektur** mit erweiterbaren Features  
- **Real-time Performance** für professionelle Anwendungen
- **Servo-Steuerung** mit präzisen UI-Kontrollen
