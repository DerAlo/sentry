#!/usr/bin/env python3
"""
Serial Controller for Arduino Communication
Handles servo control commands and Arduino communication
Port of Java LocalSerialController functionality
"""

import logging
import serial
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

class SerialController:
    """
    Arduino serial communication controller
    Sends servo control commands and receives feedback
    """
    
    def __init__(self, port: str, baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection: Optional[serial.Serial] = None
        self.connected = False
        self.running = False
        self.lock = threading.Lock()
        
    def start(self) -> bool:
        """Start serial connection"""
        try:
            logger.info(f"🔌 Connecting to Arduino on {self.port} @ {self.baudrate}")
            
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                write_timeout=1
            )
            
            # Wait for Arduino to initialize
            time.sleep(2)
            
            # Test connection
            if self.serial_connection.is_open:
                self.connected = True
                self.running = True
                
                logger.info("✅ Arduino connected successfully")
                return True
            else:
                logger.error("❌ Failed to open serial connection")
                return False
                
        except serial.SerialException as e:
            logger.error(f"❌ Serial connection error: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            self.connected = False
            return False
    
    def send_servo_command(self, x_angle: float, y_angle: float) -> bool:
        """
        Send servo control command to Arduino
        Format: "SERVO,X,Y\\n" (same as Java version)
        """
        if not self.connected or not self.serial_connection:
            logger.debug("❌ Serial not connected")
            return False
        
        try:
            with self.lock:
                # Clamp angles to valid range
                x_angle = max(0, min(180, x_angle))
                y_angle = max(0, min(180, y_angle))
                
                # Format command (same as Java)
                command = f"SERVO,{int(x_angle)},{int(y_angle)}\\n"
                
                # Send command
                self.serial_connection.write(command.encode('utf-8'))
                self.serial_connection.flush()
                
                logger.debug(f"⚙️ Sent servo command: X={x_angle}°, Y={y_angle}°")
                return True
                
        except serial.SerialException as e:
            logger.error(f"❌ Serial send error: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"❌ Send command error: {e}")
            return False
    
    def send_fire_command(self) -> bool:
        """Send fire command to Arduino"""
        if not self.connected or not self.serial_connection:
            return False
        
        try:
            with self.lock:
                command = "FIRE\\n"
                self.serial_connection.write(command.encode('utf-8'))
                self.serial_connection.flush()
                
                logger.info("🔥 Fire command sent!")
                return True
                
        except Exception as e:
            logger.error(f"❌ Fire command error: {e}")
            return False
    
    def read_response(self) -> Optional[str]:
        """Read response from Arduino"""
        if not self.connected or not self.serial_connection:
            return None
        
        try:
            if self.serial_connection.in_waiting > 0:
                response = self.serial_connection.readline().decode('utf-8').strip()
                logger.debug(f"📥 Arduino response: {response}")
                return response
        except Exception as e:
            logger.debug(f"Read error: {e}")
        
        return None
    
    def is_connected(self) -> bool:
        """Check if serial connection is active"""
        return self.connected and self.serial_connection and self.serial_connection.is_open
    
    def stop(self):
        """Close serial connection"""
        logger.info("🛑 Stopping serial controller...")
        
        self.running = False
        self.connected = False
        
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.close()
                logger.info("✅ Serial connection closed")
            except Exception as e:
                logger.error(f"Error closing serial: {e}")
        
        self.serial_connection = None

# Test the serial controller
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    controller = SerialController('/dev/ttyUSB0')  # Adjust port as needed
    
    if controller.start():
        print("Testing servo commands...")
        
        # Test servo movements
        for angle in [45, 90, 135, 90]:
            controller.send_servo_command(angle, angle)
            time.sleep(1)
        
        controller.stop()
    else:
        print("Failed to connect to Arduino")
