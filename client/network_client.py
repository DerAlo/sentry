#!/usr/bin/env python3
"""
Network Client for communicating with Raspberry Pi server
Handles video streaming and servo commands
"""

import asyncio
import json
import logging
import time
import threading
from typing import Optional, Callable, Dict, Any
import requests
import websockets
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class NetworkClient:
    """
    Network client for server communication
    Port of Java NetworkImageSource and NetworkServoController
    """
    
    def __init__(self, server_host: str = "localhost", 
                 http_port: int = 8080, websocket_port: int = 8081):
        self.server_host = server_host
        self.http_port = http_port
        self.websocket_port = websocket_port
        
        # Connection state
        self.connected = False
        self.websocket = None
        
        # Video streaming
        self.stream_url = f"http://{server_host}:{http_port}/stream"
        self.current_frame = None
        self.frame_callback: Optional[Callable] = None
        self.stream_thread = None
        self.streaming = False
        
        logger.info(f"Network client configured for {server_host}:{http_port}")
    
    async def connect_websocket(self) -> bool:
        """Connect to WebSocket servo control"""
        try:
            websocket_url = f"ws://{self.server_host}:{self.websocket_port}/servo"
            logger.info(f"Connecting to WebSocket: {websocket_url}")
            
            self.websocket = await websockets.connect(websocket_url)
            self.connected = True
            
            logger.info("✅ WebSocket connected")
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            self.connected = False
            return False
    
    async def send_servo_command(self, x_angle: float, y_angle: float) -> bool:
        """Send servo control command via WebSocket"""
        if not self.connected or not self.websocket:
            logger.warning("WebSocket not connected")
            return False
        
        try:
            command = {
                "type": "servo_control",
                "x_angle": x_angle,
                "y_angle": y_angle,
                "timestamp": time.time()
            }
            
            await self.websocket.send(json.dumps(command))
            
            # Wait for response
            response_json = await self.websocket.recv()
            response = json.loads(response_json)
            
            success = response.get('success', False)
            logger.debug(f"Servo command result: {success}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Servo command failed: {e}")
            return False
    
    def start_video_stream(self, frame_callback: Callable[[np.ndarray], None]):
        """Start receiving video stream from server"""
        self.frame_callback = frame_callback
        self.streaming = True
        
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        
        logger.info("📹 Video streaming started")
    
    def _stream_worker(self):
        """Worker thread for video streaming"""
        logger.info("📹 Video stream worker started")
        
        while self.streaming:
            try:
                # Connect to HTTP stream
                response = requests.get(self.stream_url, stream=True, timeout=5)
                
                if response.status_code != 200:
                    logger.error(f"HTTP stream error: {response.status_code}")
                    time.sleep(1)
                    continue
                
                # Parse multipart stream
                boundary = None
                buffer = b''
                
                for chunk in response.iter_content(chunk_size=1024):
                    if not self.streaming:
                        break
                    
                    buffer += chunk
                    
                    # Look for JPEG frames in the stream
                    while True:
                        # Find JPEG start
                        start = buffer.find(b'\\xff\\xd8')
                        if start == -1:
                            break
                        
                        # Find JPEG end
                        end = buffer.find(b'\\xff\\xd9', start + 2)
                        if end == -1:
                            break
                        
                        # Extract JPEG frame
                        jpeg_data = buffer[start:end + 2]
                        buffer = buffer[end + 2:]
                        
                        # Decode frame
                        try:
                            frame = cv2.imdecode(
                                np.frombuffer(jpeg_data, dtype=np.uint8), 
                                cv2.IMREAD_COLOR
                            )
                            
                            if frame is not None and self.frame_callback:
                                self.current_frame = frame
                                self.frame_callback(frame)
                                
                        except Exception as e:
                            logger.debug(f"Frame decode error: {e}")
                            continue
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Stream connection error: {e}")
                time.sleep(2)  # Wait before retry
            except Exception as e:
                logger.error(f"❌ Stream worker error: {e}")
                time.sleep(1)
        
        logger.info("📹 Video stream worker stopped")
    
    def stop_video_stream(self):
        """Stop video streaming"""
        self.streaming = False
        
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2)
        
        logger.info("📹 Video streaming stopped")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame"""
        return self.current_frame
    
    async def disconnect(self):
        """Disconnect from server"""
        self.stop_video_stream()
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.connected = False
        logger.info("🔌 Disconnected from server")
    
    def is_connected(self) -> bool:
        """Check connection status"""
        return self.connected

# Test the network client
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    async def test_client():
        client = NetworkClient()
        
        # Test WebSocket connection
        if await client.connect_websocket():
            print("✅ WebSocket connected")
            
            # Test servo commands
            print("Testing servo commands...")
            for angle in [45, 90, 135, 90]:
                success = await client.send_servo_command(angle, angle)
                print(f"Servo {angle}°: {'✅' if success else '❌'}")
                await asyncio.sleep(1)
            
            await client.disconnect()
        else:
            print("❌ WebSocket connection failed")
        
        # Test video stream
        def frame_received(frame):
            print(f"📹 Frame received: {frame.shape}")
        
        client.start_video_stream(frame_received)
        await asyncio.sleep(5)  # Stream for 5 seconds
        client.stop_video_stream()
    
    asyncio.run(test_client())
