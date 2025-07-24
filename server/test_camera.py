#!/usr/bin/env python3
"""
Simple camera test for Raspberry Pi
"""
import cv2
import sys

def test_camera():
    """Test camera access"""
    print("🧪 Testing camera access...")
    
    # Try different methods
    methods = [
        ("V4L2 Backend", cv2.CAP_V4L2),
        ("Default Backend", cv2.CAP_ANY),
        ("GSTREAMER Backend", cv2.CAP_GSTREAMER)
    ]
    
    for name, backend in methods:
        print(f"\n📹 Testing {name}...")
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                # Set basic properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"✅ {name} SUCCESS: {width}x{height} @ {fps}fps")
                    print(f"   Frame shape: {frame.shape}")
                    cap.release()
                    return True
                else:
                    print(f"❌ {name} Failed to read frame")
            else:
                print(f"❌ {name} Failed to open camera")
            cap.release()
        except Exception as e:
            print(f"❌ {name} Exception: {e}")
    
    return False

if __name__ == "__main__":
    success = test_camera()
    sys.exit(0 if success else 1)
