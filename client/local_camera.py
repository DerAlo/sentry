#!/usr/bin/env python3
"""
Local Camera Source für direkten YOLO-Betrieb
Ermöglicht Kamera-Auswahl ohne Server
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Callable, List, Dict

logger = logging.getLogger(__name__)

class LocalCameraSource:
    """
    Lokale Kamera-Quelle für direkten YOLO-Betrieb
    Alternative zum Netzwerk-Stream
    """
    
    def __init__(self):
        self.camera = None
        self.camera_index = 0
        self.is_running = False
        self.capture_thread = None
        self.frame_callback = None
        
        # Camera settings
        self.width = 640
        self.height = 480
        self.fps = 30
        
        # Available cameras
        self.available_cameras = []
        self.current_frame = None
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        logger.info("🎥 Local Camera Source initialized")
    
    def scan_cameras(self) -> List[Dict]:
        """Suche verfügbare Kameras"""
        logger.info("🔍 Scanning for available cameras...")
        cameras = []
        
        # Test ersten 10 Kamera-Indices
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Test if we can actually read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    camera_info = {
                        'index': i,
                        'name': f"Camera {i}",
                        'resolution': f"{width}x{height}",
                        'fps': fps,
                        'available': True
                    }
                    cameras.append(camera_info)
                    logger.info(f"   ✅ Found Camera {i}: {width}x{height} @ {fps:.1f}fps")
                
                cap.release()
            else:
                # Camera exists but not available
                if i == 0:  # Always include default camera
                    cameras.append({
                        'index': i,
                        'name': f"Camera {i} (Not available)",
                        'resolution': "Unknown",
                        'fps': 0,
                        'available': False
                    })
        
        self.available_cameras = cameras
        logger.info(f"🎥 Found {len([c for c in cameras if c['available']])} available cameras")
        return cameras
    
    def set_camera(self, camera_index: int) -> bool:
        """Wähle Kamera aus"""
        if self.is_running:
            self.stop_capture()
        
        self.camera_index = camera_index
        logger.info(f"🎥 Selected camera index: {camera_index}")
        return True
    
    def start_capture(self, frame_callback: Callable[[np.ndarray], None]) -> bool:
        """Starte Kamera-Aufnahme"""
        if self.is_running:
            logger.warning("Camera capture already running")
            return True
        
        self.frame_callback = frame_callback
        
        try:
            # Initialize camera
            logger.info(f"🔄 Opening camera {self.camera_index}...")
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                logger.error(f"❌ Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"✅ Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logger.info("🎥 Camera capture started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Camera initialization failed: {e}")
            return False
    
    def stop_capture(self):
        """Stoppe Kamera-Aufnahme"""
        logger.info("🛑 Stopping camera capture...")
        
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        logger.info("✅ Camera capture stopped")
    
    def _capture_loop(self):
        """Haupt-Capture-Loop"""
        logger.info("🔄 Camera capture loop started")
        
        while self.is_running and self.camera:
            try:
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    logger.warning("⚠️ Failed to read frame from camera")
                    continue
                
                # Update FPS counter
                self._update_fps()
                
                # Store current frame
                self.current_frame = frame.copy()
                
                # Call frame callback
                if self.frame_callback:
                    try:
                        self.frame_callback(frame)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")
                
                # Small delay to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                break
        
        logger.info("🛑 Camera capture loop ended")
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_fps(self) -> float:
        """Get current capture FPS"""
        return self.current_fps
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get latest captured frame"""
        return self.current_frame
    
    def is_capturing(self) -> bool:
        """Check if camera is capturing"""
        return self.is_running
    
    def get_camera_info(self) -> Dict:
        """Get current camera information"""
        if not self.camera:
            return {}
        
        return {
            'index': self.camera_index,
            'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.camera.get(cv2.CAP_PROP_FPS),
            'actual_fps': self.current_fps,
            'is_capturing': self.is_running
        }
    
    def set_resolution(self, width: int, height: int):
        """Set camera resolution"""
        self.width = width
        self.height = height
        
        if self.camera and self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            logger.info(f"📐 Resolution set to {width}x{height}")
    
    def set_fps(self, fps: int):
        """Set camera FPS"""
        self.fps = fps
        
        if self.camera and self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            logger.info(f"🎬 FPS set to {fps}")

def main():
    """Test der Local Camera Source"""
    print("🎥 Local Camera Source Test")
    print("=" * 30)
    
    camera_source = LocalCameraSource()
    
    # Scan for cameras
    cameras = camera_source.scan_cameras()
    
    if not cameras:
        print("❌ No cameras found")
        return
    
    print("\n📋 Available Cameras:")
    for cam in cameras:
        status = "✅" if cam['available'] else "❌"
        print(f"   {status} {cam['name']}: {cam['resolution']} @ {cam['fps']:.1f}fps")
    
    # Use first available camera
    available_cams = [c for c in cameras if c['available']]
    if available_cams:
        selected_cam = available_cams[0]
        print(f"\n🎥 Testing camera {selected_cam['index']}...")
        
        def frame_callback(frame):
            print(f"📸 Frame received: {frame.shape}")
        
        camera_source.set_camera(selected_cam['index'])
        if camera_source.start_capture(frame_callback):
            print("✅ Camera test successful!")
            time.sleep(3)
            camera_source.stop_capture()
        else:
            print("❌ Camera test failed")

if __name__ == '__main__':
    main()
