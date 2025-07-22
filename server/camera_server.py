#!/usr/bin/env python3
"""
Cat Tracking Server - Raspberry Pi
Provides webcam streaming and servo control via WebSocket
Port of the Java NetworkImageSource functionality with same API
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

import cv2
import websockets
from flask import Flask, Response, jsonify
import threading
import queue

from serial_controller import SerialController

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class CameraServer:
    """
    High-performance camera server with HTTP streaming and WebSocket servo control
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.flask_app = Flask(__name__)
        self.setup_routes()
        
        # Camera setup
        self.camera = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.current_frame = None
        self.camera_thread = None
        self.running = False
        
        # Serial controller for Arduino
        self.serial_controller = SerialController(
            port=config.get('serial_port', '/dev/ttyUSB0'),
            baudrate=config.get('serial_baudrate', 9600)
        )
        
        logger.info("🚀 Cat Tracking Server initialized")
    
    def setup_routes(self):
        """Setup Flask HTTP routes for camera streaming"""
        
        @self.flask_app.route('/')
        def index():
            return jsonify({
                "service": "Cat Tracking Camera Server",
                "status": "running",
                "endpoints": {
                    "stream": "/stream",
                    "websocket": f"ws://localhost:{self.config.get('websocket_port', 8081)}/servo"
                }
            })
        
        @self.flask_app.route('/stream')
        def video_stream():
            """HTTP streaming endpoint - same as Java version"""
            return Response(
                self.generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
    
    def init_camera(self) -> bool:
        """Initialize camera with optimized settings"""
        try:
            camera_device = self.config.get('camera_device', 0)
            logger.info(f"📹 Initializing camera device: {camera_device}")
            
            self.camera = cv2.VideoCapture(camera_device)
            
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera device: {camera_device}")
                return False
            
            # Optimize camera settings for performance
            width = self.config.get('camera_width', 640)
            height = self.config.get('camera_height', 480)
            fps = self.config.get('camera_fps', 30)
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Low latency
            
            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"📹 Camera connected: {actual_width}x{actual_height} @ {actual_fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def capture_frames(self):
        """Camera capture thread - high performance frame capture"""
        logger.info("📹 Camera capture thread started")
        
        while self.running:
            try:
                if self.camera is None or not self.camera.isOpened():
                    time.sleep(1)
                    continue
                
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Store current frame
                self.current_frame = frame.copy()
                
                # Add to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
                
            except Exception as e:
                logger.error(f"Frame capture error: {e}")
                time.sleep(0.5)
        
        logger.info("📹 Camera capture thread stopped")
    
    def generate_frames(self):
        """Generate frames for HTTP streaming - same format as Java"""
        while True:
            try:
                if self.current_frame is None:
                    time.sleep(0.1)
                    continue
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', self.current_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                # HTTP streaming format (same as Java)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except Exception as e:
                logger.error(f"Frame generation error: {e}")
                time.sleep(0.1)
    
    async def handle_websocket_servo(self, websocket, path):
        """Handle WebSocket servo control commands"""
        logger.info(f"🔌 WebSocket client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    command = json.loads(message)
                    await self.process_servo_command(command, websocket)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "error": "Invalid JSON format"
                    }))
                except Exception as e:
                    logger.error(f"WebSocket message error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    async def process_servo_command(self, command: Dict[str, Any], websocket):
        """Process servo control commands - same API as Java"""
        try:
            cmd_type = command.get('type')
            
            if cmd_type == 'servo_control':
                x_angle = command.get('x_angle', 90)
                y_angle = command.get('y_angle', 90)
                
                # Send to Arduino via serial
                success = self.serial_controller.send_servo_command(x_angle, y_angle)
                
                response = {
                    "type": "servo_response",
                    "success": success,
                    "x_angle": x_angle,
                    "y_angle": y_angle,
                    "timestamp": time.time()
                }
                
                await websocket.send(json.dumps(response))
                logger.debug(f"⚙️ Servo command: X={x_angle}°, Y={y_angle}° -> {success}")
                
            elif cmd_type == 'ping':
                await websocket.send(json.dumps({
                    "type": "pong",
                    "timestamp": time.time()
                }))
            
            else:
                await websocket.send(json.dumps({
                    "error": f"Unknown command type: {cmd_type}"
                }))
                
        except Exception as e:
            logger.error(f"Command processing error: {e}")
            await websocket.send(json.dumps({
                "error": str(e)
            }))
    
    def start(self):
        """Start the server (HTTP + WebSocket)"""
        try:
            # Initialize camera
            if not self.init_camera():
                logger.error("❌ Failed to initialize camera")
                return False
            
            # Start serial controller
            self.serial_controller.start()
            
            # Start camera capture thread
            self.running = True
            self.camera_thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.camera_thread.start()
            
            # Start WebSocket server
            websocket_port = self.config.get('websocket_port', 8081)
            websocket_server = websockets.serve(
                self.handle_websocket_servo, 
                'localhost', 
                websocket_port
            )
            
            # Start Flask in a separate thread
            http_port = self.config.get('http_port', 8080)
            flask_thread = threading.Thread(
                target=lambda: self.flask_app.run(
                    host='0.0.0.0', 
                    port=http_port, 
                    debug=False,
                    threaded=True
                ),
                daemon=True
            )
            flask_thread.start()
            
            logger.info(f"🚀 Server running:")
            logger.info(f"   📹 HTTP Stream: http://localhost:{http_port}/stream")
            logger.info(f"   🔌 WebSocket: ws://localhost:{websocket_port}/servo")
            
            # Run WebSocket server
            asyncio.get_event_loop().run_until_complete(websocket_server)
            asyncio.get_event_loop().run_forever()
            
        except KeyboardInterrupt:
            logger.info("🛑 Server shutdown requested")
            self.stop()
        except Exception as e:
            logger.error(f"Server error: {e}")
            return False
    
    def stop(self):
        """Clean shutdown"""
        logger.info("🛑 Stopping server...")
        
        self.running = False
        
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)
        
        if self.camera:
            self.camera.release()
        
        self.serial_controller.stop()
        logger.info("✅ Server stopped")

def load_config() -> Dict[str, Any]:
    """Load server configuration"""
    config = {
        # Camera settings
        'camera_device': 0,
        'camera_width': 640,
        'camera_height': 480,
        'camera_fps': 30,
        
        # Network settings
        'http_port': 8080,
        'websocket_port': 8081,
        
        # Serial settings
        'serial_port': '/dev/ttyUSB0',  # Linux/RPi
        'serial_baudrate': 9600
    }
    
    # Try to load from config file
    config_file = Path('server_config.json')
    if config_file.exists():
        try:
            with open(config_file) as f:
                file_config = json.load(f)
                config.update(file_config)
                logger.info(f"📁 Config loaded from {config_file}")
        except Exception as e:
            logger.warning(f"Config file error: {e}")
    
    return config

if __name__ == '__main__':
    print("🐱 Cat Tracking Server - Python Edition")
    print("🚀 Starting server with CUDA-ready client support...")
    
    config = load_config()
    server = CameraServer(config)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")
    finally:
        server.stop()
