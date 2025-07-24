#!/usr/bin/env python3
"""
Cat Tracking GUI Client - Python Edition mit CUDA
Enhanced version with all Java features:
- Friend/Enemy dropdown selection
- Fire enable checkbox
- Video recording when engaging
- Servo reset button
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
from PIL import Image, ImageTk

from yolo_tracker import CudaYoloTracker
from network_client import NetworkClient
from local_camera import LocalCameraSource
from coco_classes import COCO_CLASSES, DEFAULT_FRIENDS, DEFAULT_ENEMIES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class CatTrackingGUI:
    """
    Enhanced Cat Tracking GUI with all Java features
    Performance-Ziel: 30-60 FPS (vs 5-10 FPS Java)
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cat Tracking System - Python CUDA Edition (Enhanced)")
        self.root.geometry("1400x900")
        
        # Components
        self.yolo_tracker = CudaYoloTracker()
        self.network_client = None  # Will be created dynamically with user IP
        self.local_camera = LocalCameraSource()
        
        # Mode selection
        self.use_local_camera = False
        
        # State
        self.current_frame = None
        self.tracking_enabled = False
        self.overlays_enabled = True
        self.target_object = None
        self.detections = []
        
        # Performance tracking
        self.fps_display = 0.0
        self.yolo_fps = 0.0
        
        # Friend/Enemy classification (configurable like Java)
        self.friend_classes = set(DEFAULT_FRIENDS)  # person = friend by default  
        self.enemy_classes = set(DEFAULT_ENEMIES)   # cat = enemy by default
        
        # Tracking state for logging optimization
        self.last_detection_state = None  # Track detection changes
        self.last_status_message = ""     # Avoid duplicate status messages
        self.last_servo_command_time = 0  # Track servo command timing
        self.servo_center_reported = False  # Track if center position was reported
        
        # Combat controls (like Java version)
        self.fire_enabled = tk.BooleanVar(value=False)
        self.video_recording = tk.BooleanVar(value=False)
        self.video_writer = None
        
        # Setup UI FIRST
        self.setup_ui()
        
        # Initialize classification display after UI is ready
        self.update_classification_display()
        
        # Load YOLO model AFTER GUI is complete
        self.load_yolo_model()
        
        logger.info("🚀 Enhanced Cat Tracking GUI initialized")
    
    def is_network_connected(self):
        """Helper to check if network client is connected"""
        return self.network_client is not None and self.network_client.is_connected()
    
    def setup_ui(self):
        """Setup the enhanced user interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video display
        self.setup_video_display(main_frame)
        
        # Enhanced control panel
        self.setup_control_panel(main_frame)
        
        # Status bar
        self.setup_status_bar()
    
    def setup_video_display(self, parent):
        """Setup video display area"""
        video_frame = ttk.LabelFrame(parent, text="📹 Live Video Feed")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Video canvas
        self.video_canvas = tk.Canvas(video_frame, bg='black', width=640, height=480)
        self.video_canvas.pack(padx=5, pady=5)
        
        # Video controls
        video_controls = ttk.Frame(video_frame)
        video_controls.pack(fill=tk.X, padx=5, pady=5)
        
        # Source selection (Network vs Local)
        source_frame = ttk.LabelFrame(video_controls, text="📹 Video Source")
        source_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.source_var = tk.StringVar(value="network")
        
        network_radio = ttk.Radiobutton(source_frame, text="🌐 Network (Server)",
                                       variable=self.source_var, value="network",
                                       command=self.on_source_change)
        network_radio.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Server IP input
        ip_frame = ttk.Frame(source_frame)
        ip_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(ip_frame, text="Server IP:").pack(side=tk.LEFT)
        self.server_ip_var = tk.StringVar(value="pi3b")
        self.server_ip_entry = ttk.Entry(ip_frame, textvariable=self.server_ip_var, width=15)
        self.server_ip_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        local_radio = ttk.Radiobutton(source_frame, text="📹 Local Camera",
                                     variable=self.source_var, value="local",
                                     command=self.on_source_change)
        local_radio.pack(side=tk.LEFT, padx=(10, 5), pady=5)
        
        # Local camera controls (initially hidden)
        self.camera_frame = ttk.Frame(source_frame)
        
        self.camera_combo = ttk.Combobox(self.camera_frame, width=20)
        self.camera_combo.pack(side=tk.LEFT, padx=5)
        self.camera_combo.bind('<<ComboboxSelected>>', self.on_camera_select)
        
        ttk.Button(self.camera_frame, text="🔍 Scan Cameras",
                  command=self.scan_cameras).pack(side=tk.LEFT, padx=5)
        
        # Main controls
        controls_frame = ttk.Frame(video_controls)
        controls_frame.pack(fill=tk.X, pady=5)
        
        self.connect_btn = ttk.Button(controls_frame, text="🔌 Connect to Server",
                                     command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT)
        
        self.tracking_btn = ttk.Button(controls_frame, text="🎯 Start Tracking",
                                      command=self.toggle_tracking)
        self.tracking_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        self.overlays_btn = ttk.Button(controls_frame, text="🎨 Toggle Overlays",
                                      command=self.toggle_overlays)
        self.overlays_btn.pack(side=tk.LEFT)
    
    def setup_control_panel(self, parent):
        """Setup enhanced control panel"""
        control_frame = ttk.LabelFrame(parent, text="⚙️ Enhanced Control Panel")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # CUDA Status
        cuda_frame = ttk.LabelFrame(control_frame, text="🚀 CUDA Status")
        cuda_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.cuda_status = ttk.Label(cuda_frame, text="Initializing...")
        self.cuda_status.pack(padx=5, pady=5)
        
        # Performance
        perf_frame = ttk.LabelFrame(control_frame, text="📊 Performance")
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.fps_label = ttk.Label(perf_frame, text="FPS: --")
        self.fps_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.yolo_fps_label = ttk.Label(perf_frame, text="YOLO FPS: --")
        self.yolo_fps_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.detections_label = ttk.Label(perf_frame, text="Objects: 0")
        self.detections_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Combat Settings (like Java version)
        combat_frame = ttk.LabelFrame(control_frame, text="⚔️ Combat Settings")
        combat_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Fire Enable Checkbox
        self.fire_checkbox = ttk.Checkbutton(combat_frame, text="🔥 SCHIESSBEFEHL aktiviert", 
                                           variable=self.fire_enabled,
                                           command=self.on_fire_enable_change)
        self.fire_checkbox.pack(anchor=tk.W, padx=5, pady=5)
        
        # Video Recording Checkbox
        self.video_checkbox = ttk.Checkbutton(combat_frame, text="📹 Video bei Feindkontakt", 
                                            variable=self.video_recording)
        self.video_checkbox.pack(anchor=tk.W, padx=5, pady=2)
        
        # Friend/Enemy Classification (like Java version)
        classification_frame = ttk.LabelFrame(control_frame, text="👥 Freund-Feind-Erkennung")
        classification_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Friend selection
        friend_frame = ttk.Frame(classification_frame)
        friend_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(friend_frame, text="👥 Freunde:").pack(anchor=tk.W)
        
        friend_control_frame = ttk.Frame(friend_frame)
        friend_control_frame.pack(fill=tk.X, pady=2)
        
        self.friend_combo = ttk.Combobox(friend_control_frame, 
                                       values=[f"{i}: {cls}" for i, cls in enumerate(COCO_CLASSES)],
                                       width=20)
        self.friend_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(friend_control_frame, text="➕ Add",
                  command=self.add_friend, width=8).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(friend_control_frame, text="➖ Remove", 
                  command=self.remove_friend, width=8).pack(side=tk.LEFT)
        
        # Enemy selection
        enemy_frame = ttk.Frame(classification_frame)
        enemy_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(enemy_frame, text="👤 Feinde:").pack(anchor=tk.W)
        
        enemy_control_frame = ttk.Frame(enemy_frame)
        enemy_control_frame.pack(fill=tk.X, pady=2)
        
        self.enemy_combo = ttk.Combobox(enemy_control_frame, 
                                      values=[f"{i}: {cls}" for i, cls in enumerate(COCO_CLASSES)],
                                      width=20)
        self.enemy_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(enemy_control_frame, text="➕ Add",
                  command=self.add_enemy, width=8).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(enemy_control_frame, text="➖ Remove",
                  command=self.remove_enemy, width=8).pack(side=tk.LEFT)
        
        # Current classification display
        self.classification_display = tk.Text(classification_frame, height=4, width=30)
        self.classification_display.pack(fill=tk.X, padx=5, pady=5)
        self.update_classification_display()
        
        # Servo Control
        servo_frame = ttk.LabelFrame(control_frame, text="🕹️ Manual Servo Control")
        servo_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # X Servo
        ttk.Label(servo_frame, text="X Servo:").pack(anchor=tk.W, padx=5)
        self.x_servo_var = tk.DoubleVar(value=90)
        self.x_servo_scale = ttk.Scale(servo_frame, from_=0, to=180, 
                                      variable=self.x_servo_var,
                                      orient=tk.HORIZONTAL,
                                      command=self.on_servo_change)
        self.x_servo_scale.pack(fill=tk.X, padx=5, pady=2)
        self.x_servo_label = ttk.Label(servo_frame, text="90°")
        self.x_servo_label.pack(anchor=tk.W, padx=5)
        
        # Y Servo
        ttk.Label(servo_frame, text="Y Servo:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.y_servo_var = tk.DoubleVar(value=90)
        self.y_servo_scale = ttk.Scale(servo_frame, from_=0, to=180,
                                      variable=self.y_servo_var,
                                      orient=tk.HORIZONTAL,
                                      command=self.on_servo_change)
        self.y_servo_scale.pack(fill=tk.X, padx=5, pady=2)
        self.y_servo_label = ttk.Label(servo_frame, text="90°")
        self.y_servo_label.pack(anchor=tk.W, padx=5)
        
        # Servo buttons
        servo_buttons = ttk.Frame(servo_frame)
        servo_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        # Reset Servos button (like Java version)
        self.reset_servos_btn = ttk.Button(servo_buttons, text="🎯 Reset Servos", 
                                          command=self.reset_servos)
        self.reset_servos_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Fire button (initially disabled)
        self.fire_btn = ttk.Button(servo_buttons, text="🔥 FIRE!", 
                                  command=self.fire_command,
                                  state='disabled')
        self.fire_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Target Selection
        target_frame = ttk.LabelFrame(control_frame, text="🎯 Target Selection")
        target_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.target_listbox = tk.Listbox(target_frame, height=6)
        self.target_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.target_listbox.bind('<<ListboxSelect>>', self.on_target_select)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # [Continue with all the methods from the previous implementation...]
    # I'll add the essential methods here:
    
    def load_yolo_model(self):
        """Load YOLO model in background"""
        def load_worker():
            try:
                self.root.after(0, lambda: self.update_status("Loading CUDA YOLO model..."))
                
                if self.yolo_tracker.load_model():
                    self.root.after(0, lambda: self.update_status("YOLO model loaded successfully"))
                    
                    # Update CUDA status - always try to update if possible
                    if self.yolo_tracker.device.type == 'cuda':
                        cuda_info = f"✅ CUDA: {self.yolo_tracker.device}"
                    else:
                        cuda_info = "⚠️ CPU Only"
                    
                    # Try to update CUDA status (will be ignored if element doesn't exist)
                    try:
                        self.root.after(0, lambda: self.cuda_status.config(text=cuda_info))
                    except AttributeError:
                        pass  # Element doesn't exist yet, ignore
                    
                    # Try TensorRT optimization
                    self.root.after(0, lambda: self.update_status("Optimizing with TensorRT..."))
                    self.yolo_tracker.optimize_with_tensorrt()
                    self.root.after(0, lambda: self.update_status("TensorRT optimization complete"))
                    
                else:
                    self.root.after(0, lambda: self.update_status("❌ Failed to load YOLO model"))
                    
            except Exception as e:
                logger.error(f"Model loading error: {e}")
                self.root.after(0, lambda: self.update_status(f"❌ Model error: {e}"))
        
        threading.Thread(target=load_worker, daemon=True).start()
    
    def fire_command(self):
        """Send fire command (only if enabled like Java version)"""
        if not self.fire_enabled.get():
            self.update_status("🚫 Schießbefehl ist deaktiviert!")
            messagebox.showwarning("Nicht aktiviert", "SCHIESSBEFEHL muss aktiviert sein!")
            return
            
        if self.use_local_camera:
            # Local mode: simulate fire command
            self.update_status("🔥 FIRE! (Simulation)")
            messagebox.showinfo("Fire!", "Feuerbefehl ausgeführt! (Simulation)")
        elif self.network_client and self.network_client.is_connected():
            # Network mode: send real fire command
            self.update_status("🔥 FIRE! (Real)")
            messagebox.showinfo("Fire!", "Feuerbefehl gesendet!")
            # Would send actual fire command via WebSocket
        else:
            messagebox.showwarning("Not Connected", "Nicht mit Server verbunden!")
    
    def reset_servos(self):
        """Reset servos to center position (like Java version)"""
        self.x_servo_var.set(90)
        self.y_servo_var.set(90)
        self.update_status("🎯 Servos zurückgesetzt")
        
        # Send reset command if connected
        if self.network_client and self.network_client.is_connected():
            task = asyncio.create_task(self.send_servo_async(90, 90))
            self._pending_tasks = getattr(self, '_pending_tasks', [])
            self._pending_tasks.append(task)
    
    def on_fire_enable_change(self):
        """Handle fire enable checkbox change"""
        if self.fire_enabled.get():
            self.update_status("⚔️ SCHIESSBEFEHL AKTIVIERT!")
            self.fire_btn.config(state='normal')
            # Update display
            self.update_classification_display()
        else:
            self.update_status("🛡️ Schießbefehl deaktiviert")
            self.fire_btn.config(state='disabled')
            self.update_classification_display()
    
    def add_friend(self):
        """Add selected class as friend"""
        selection = self.friend_combo.get()
        if selection:
            try:
                class_id = int(selection.split(':')[0])
                class_name = selection.split(': ')[1]
                self.friend_classes.add(class_id)
                self.enemy_classes.discard(class_id)  # Remove from enemies
                self.update_classification_display()
                self.update_status(f"👥 {class_name} als Freund hinzugefügt")
            except (ValueError, IndexError):
                messagebox.showerror("Fehler", "Ungültige Auswahl")
    
    def add_enemy(self):
        """Add selected class as enemy"""
        selection = self.enemy_combo.get()
        if selection:
            try:
                class_id = int(selection.split(':')[0])
                class_name = selection.split(': ')[1]
                self.enemy_classes.add(class_id)
                self.friend_classes.discard(class_id)  # Remove from friends
                self.update_classification_display()
                self.update_status(f"👤 {class_name} als Feind hinzugefügt")
            except (ValueError, IndexError):
                messagebox.showerror("Fehler", "Ungültige Auswahl")
    
    def update_classification_display(self):
        """Update the classification display text"""
        self.classification_display.delete(1.0, tk.END)
        
        # Friends
        friend_names = [COCO_CLASSES[i] for i in self.friend_classes]
        self.classification_display.insert(tk.END, f"👥 Freunde: {', '.join(friend_names)}\n")
        
        # Enemies  
        enemy_names = [COCO_CLASSES[i] for i in self.enemy_classes]
        self.classification_display.insert(tk.END, f"👤 Feinde: {', '.join(enemy_names)}\n")
        
        self.classification_display.insert(tk.END, f"⚔️ Feuer: {'✅' if self.fire_enabled.get() else '❌'}\n")
        self.classification_display.insert(tk.END, f"📹 Video: {'✅' if self.video_recording.get() else '❌'}")
    
    def toggle_tracking(self):
        """Toggle object tracking"""
        self.tracking_enabled = not self.tracking_enabled
        
        if self.tracking_enabled:
            self.tracking_btn.config(text="🛑 Stop Tracking")
            self.update_status("Object tracking enabled")
        else:
            self.tracking_btn.config(text="🎯 Start Tracking")
            self.target_object = None
            self.update_status("Object tracking disabled")
    
    def toggle_overlays(self):
        """Toggle YOLO overlays"""
        self.overlays_enabled = not self.overlays_enabled
        status = "enabled" if self.overlays_enabled else "disabled"
        self.update_status(f"Overlays {status}")
    
    def update_status(self, message: str, force_log: bool = False):
        """Update status bar with smart logging (avoid spam)"""
        # Only log if message changed or forced
        if message != self.last_status_message or force_log:
            # Update status bar - always try to update
            try:
                self.status_bar.config(text=message)
            except AttributeError:
                pass  # Status bar not initialized yet
            if force_log or "FIRE" in message or "LOCKED" in message or "RESET" in message:
                logger.info(message)
            self.last_status_message = message
    
    # Missing methods from original implementation
    def on_source_change(self):
        """Handle video source change"""
        source = self.source_var.get()
        
        if source == "local":
            self.use_local_camera = True
            self.camera_frame.pack(side=tk.LEFT, padx=(10, 0))
            self.connect_btn.config(text="🎥 Start Camera")
            self.update_status("📹 Switched to local camera mode")
        else:
            self.use_local_camera = False
            self.camera_frame.pack_forget()
            self.connect_btn.config(text="🔌 Connect to Server")
            self.update_status("🌐 Switched to network mode")
            
            # Stop local camera if running
            if self.local_camera.is_capturing():
                self.local_camera.stop_capture()
    
    def toggle_connection(self):
        """Toggle server connection or local camera"""
        if self.use_local_camera:
            # Local camera mode
            if self.local_camera.is_capturing():
                # Stop camera
                self.local_camera.stop_capture()
                self.connect_btn.config(text="🎥 Start Camera")
                self.update_status("📹 Camera stopped")
            else:
                # Start camera
                if self.local_camera.start_capture(self.on_frame_received):
                    self.connect_btn.config(text="🛑 Stop Camera")
                    self.update_status("✅ Camera started")
                else:
                    self.update_status("❌ Failed to start camera")
        else:
            # Network mode (original logic)
            if self.network_client and self.network_client.is_connected():
                # Disconnect
                task = asyncio.create_task(self.disconnect_server())
                # Keep reference to prevent garbage collection
                self._pending_tasks = getattr(self, '_pending_tasks', [])
                self._pending_tasks.append(task)
            else:
                # Connect
                threading.Thread(target=self.connect_server, daemon=True).start()
    
    def scan_cameras(self):
        """Scan for available cameras"""
        def scan_worker():
            try:
                self.update_status("🔍 Scanning for cameras...")
                cameras = self.local_camera.scan_cameras()
                
                # Update combobox
                camera_list = []
                for cam in cameras:
                    if cam['available']:
                        camera_list.append(f"Camera {cam['index']} ({cam['resolution']})")
                    else:
                        camera_list.append(f"Camera {cam['index']} (Not available)")
                
                self.root.after(0, lambda: self.camera_combo.config(values=camera_list))
                if camera_list:
                    self.root.after(0, lambda: self.camera_combo.set(camera_list[0]))
                
                self.root.after(0, lambda: self.update_status(f"✅ Found {len([c for c in cameras if c['available']])} cameras"))
                
            except Exception as e:
                logger.error(f"Camera scan error: {e}")
                self.root.after(0, lambda: self.update_status(f"❌ Camera scan error: {e}"))
        
        threading.Thread(target=scan_worker, daemon=True).start()
    
    def on_camera_select(self, event=None):
        """Handle camera selection"""
        selection = self.camera_combo.get()
        if selection:
            # Extract camera index
            try:
                camera_index = int(selection.split()[1])
                self.local_camera.set_camera(camera_index)
                self.update_status(f"📹 Selected camera {camera_index}")
            except (ValueError, IndexError):
                logger.error("Invalid camera selection")
    
    def connect_server(self):
        """Connect to server in background"""
        async def connect_worker():
            try:
                # Get server IP from input field
                server_ip = self.server_ip_var.get().strip()
                if not server_ip:
                    self.root.after(0, lambda: self.update_status("❌ Please enter server IP"))
                    return
                
                # Create new network client with user IP
                self.network_client = NetworkClient(server_host=server_ip, http_port=8080, websocket_port=8081)
                
                self.root.after(0, lambda: self.update_status(f"Connecting to {server_ip}..."))
                
                # Connect WebSocket
                if await self.network_client.connect_websocket():
                    # Start video stream
                    self.network_client.start_video_stream(self.on_frame_received)
                    
                    self.root.after(0, lambda: self.connect_btn.config(text="🔌 Disconnect"))
                    self.root.after(0, lambda: self.update_status(f"✅ Connected to {server_ip}"))
                else:
                    self.root.after(0, lambda: self.update_status("❌ Failed to connect"))
                    
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.root.after(0, lambda: self.update_status(f"❌ Connection error: {e}"))
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(connect_worker())
        loop.close()
    
    async def disconnect_server(self):
        """Disconnect from server"""
        await self.network_client.disconnect()
        self.connect_btn.config(text="🔌 Connect to Server")
        self.update_status("Disconnected from server")
    
    def on_frame_received(self, frame: np.ndarray):
        """Handle received video frame"""
        logger.info(f"📹 Frame received: {frame.shape if frame is not None else 'None'}")
        self.current_frame = frame
        
        # Run YOLO detection if tracking enabled
        if self.tracking_enabled and self.yolo_tracker.model is not None:
            self.detections = self.yolo_tracker.detect_objects(frame)
            self.yolo_fps = self.yolo_tracker.get_fps()
            
            # Enhanced auto-targeting logic with video recording
            self.process_auto_targeting()
        
        # Update display
        self.root.after(0, self.update_video_display)
    
    def process_auto_targeting(self):
        """Process enhanced auto-targeting logic with smart logging"""
        current_state = {
            'friends': len([d for d in self.detections if d['class_id'] in self.friend_classes]),
            'enemies': len([d for d in self.detections if d['class_id'] in self.enemy_classes]),
            'total': len(self.detections)
        }
        
        # Check if detection state changed (to avoid spam)
        state_changed = current_state != self.last_detection_state
        self.last_detection_state = current_state.copy()
        
        if not self.detections:
            # Stop video recording if no objects detected
            if self.video_writer is not None:
                self.stop_video_recording()
            
            # Reset servo centering flag when no detections
            self.servo_center_reported = False
            return
        
        # Check for friends (hold fire like Java)
        friends = [d for d in self.detections if d['class_id'] in self.friend_classes]
        enemies = [d for d in self.detections if d['class_id'] in self.enemy_classes]
        
        if friends:
            # Friends present - hold fire (like Java)
            if state_changed:  # Only log when state changes
                friend_names = [d['class_name'] for d in friends]
                self.update_status(f"👥 FRIEND DETECTED: {', '.join(set(friend_names))} - HOLD FIRE", force_log=True)
            else:
                self.update_status("👥 FRIEND IN ZONE / HOLD FIRE")
            
            if self.video_writer is not None:
                self.stop_video_recording()
            return
        elif enemies:
            # Enemy only - armed (like Java)
            if state_changed:  # Only log when enemies first detected
                enemy_names = [d['class_name'] for d in enemies]
                self.update_status(f"👤 ENEMY DETECTED: {', '.join(set(enemy_names))} - ARMED", force_log=True)
            else:
                self.update_status("👤 ENEMY ONLY / ARMED")
            
            # Start video recording if enabled (like Java)
            if self.video_recording.get() and not self.video_writer:
                self.start_video_recording()
            
            # Record current frame
            self.record_frame()
            
            # Target enemy with highest confidence
            target = max(enemies, key=lambda x: x['confidence'])
            self.target_object = target
            
            # Calculate servo angles for targeting (same as Java)
            frame_center_x = self.current_frame.shape[1] // 2
            frame_center_y = self.current_frame.shape[0] // 2
            
            bbox = target['bbox']
            target_x = bbox['x'] + bbox['width'] // 2
            target_y = bbox['y'] + bbox['height'] // 2
            
            # Convert to servo angles (same logic as Java)
            x_angle = 90 + (target_x - frame_center_x) * 0.1
            y_angle = 90 + (target_y - frame_center_y) * 0.1
            
            # Clamp angles (like Java servo limits)
            x_angle = max(0, min(180, x_angle))
            y_angle = max(0, min(180, y_angle))
            
            # Check if aimed and high confidence (like Java >= 0.70)
            confidence = target['confidence']
            pos_tolerance = 30  # Like Java
            
            aimed = (abs(target_x - frame_center_x) <= pos_tolerance and 
                    abs(target_y - frame_center_y) <= pos_tolerance)
            
            if confidence >= 0.70 and aimed and self.fire_enabled.get():
                # HIGH CONFIDENCE TARGET ACQUIRED (like Java) - Always log fire events
                self.update_status(f"🎯 LOCKED ON: {target['class_name']} ({confidence:.1%}) - ENGAGING!", force_log=True)
                
                # Record the engagement
                self.record_frame()
                
                # Automatic fire (like Java debouncer logic)
                if hasattr(self, '_last_fire_time'):
                    if time.time() - self._last_fire_time > 2.0:  # 2 second cooldown
                        self._last_fire_time = time.time()
                        self.fire_command()
                else:
                    self._last_fire_time = time.time()
                    self.fire_command()
            else:
                # Send servo command for tracking (throttled logging)
                current_time = time.time()
                if self.network_client.is_connected():
                    task = asyncio.create_task(self.send_servo_async(x_angle, y_angle))
                    self._pending_tasks = getattr(self, '_pending_tasks', [])
                    self._pending_tasks.append(task)
                else:
                    # Local mode: simulate servo commands (throttled)
                    self.simulate_servo_commands(x_angle, y_angle, current_time)
        
        # Reset servo centering flag when actively tracking
        self.servo_center_reported = False
    
    async def send_servo_async(self, x_angle: float, y_angle: float):
        """Send servo command asynchronously"""
        try:
            await self.network_client.send_servo_command(x_angle, y_angle)
        except Exception as e:
            logger.error(f"Servo command error: {e}")
    
    def simulate_servo_commands(self, x_angle: float, y_angle: float, current_time: float = None):
        """Simulate servo commands in local mode by updating sliders (throttled logging)"""
        # Update servo sliders to show tracking movement
        self.x_servo_var.set(x_angle)
        self.y_servo_var.set(y_angle)
        
        # Update labels
        self.x_servo_label.config(text=f"{x_angle:.1f}°")
        self.y_servo_label.config(text=f"{y_angle:.1f}°")
        
        # Show tracking status (throttled - only every 2 seconds or on significant change)
        if current_time is None:
            current_time = time.time()
            
        if (current_time - self.last_servo_command_time > 2.0 or 
            abs(x_angle - 90) < 5 and abs(y_angle - 90) < 5):  # Near center or timeout
            
            if self.target_object:
                class_name = self.target_object['class_name']
                confidence = self.target_object['confidence']
                
                # Check if servos are near center (auto-centered)
                if abs(x_angle - 90) < 5 and abs(y_angle - 90) < 5 and not self.servo_center_reported:
                    self.update_status(f"🎯 Target lost: {class_name} - Servos centered", force_log=True)
                    self.servo_center_reported = True
                else:
                    self.update_status(f"🎯 Tracking {class_name} ({confidence:.1%}) - X={x_angle:.0f}° Y={y_angle:.0f}°")
                    
            self.last_servo_command_time = current_time
    
    def update_video_display(self):
        """Update video display with overlays"""
        if self.current_frame is None:
            return
        
        frame = self.current_frame.copy()
        
        # Draw YOLO overlays
        if self.overlays_enabled and self.detections:
            self.draw_yolo_overlays(frame)
        
        # Convert to PhotoImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Resize to fit canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            image = image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image)
        
        # Update canvas
        self.video_canvas.delete("all")
        self.video_canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
        self.video_canvas.image = photo  # Keep reference
        
        # Update performance labels
        self.fps_label.config(text=f"FPS: {self.fps_display:.1f}")
        self.yolo_fps_label.config(text=f"YOLO FPS: {self.yolo_fps:.1f}")
        self.detections_label.config(text=f"Objects: {len(self.detections)}")
        
        # Update target list
        self.update_target_list()
    
    def draw_yolo_overlays(self, frame: np.ndarray):
        """Draw YOLO detection overlays with friend/enemy colors"""
        for detection in self.detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            # Determine color based on friend/enemy (like Java)
            if class_id in self.friend_classes:
                color = (0, 255, 0)  # Green for friends
            elif class_id in self.enemy_classes:
                color = (0, 0, 255)  # Red for enemies
            else:
                color = (255, 255, 0)  # Yellow for unknown
            
            # Draw bounding box
            cv2.rectangle(frame, 
                         (bbox['x'], bbox['y']),
                         (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                         color, 2)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text
            cv2.rectangle(frame,
                         (bbox['x'], bbox['y'] - label_size[1] - 10),
                         (bbox['x'] + label_size[0], bbox['y']),
                         color, -1)
            
            # Text
            cv2.putText(frame, label, (bbox['x'], bbox['y'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Target marker
            if detection == self.target_object:
                center_x = bbox['x'] + bbox['width'] // 2
                center_y = bbox['y'] + bbox['height'] // 2
                cv2.drawMarker(frame, (center_x, center_y), (255, 0, 255), 
                              cv2.MARKER_CROSS, 20, 3)
    
    def update_target_list(self):
        """Update target selection list"""
        self.target_listbox.delete(0, tk.END)
        
        for i, detection in enumerate(self.detections):
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # Add type indicator
            if class_id in self.friend_classes:
                type_icon = "👥"
            elif class_id in self.enemy_classes:
                type_icon = "👤"
            else:
                type_icon = "❓"
            
            label = f"{type_icon} {class_name} ({confidence:.1%})"
            self.target_listbox.insert(tk.END, label)
    
    def on_target_select(self, event):
        """Handle target selection"""
        selection = self.target_listbox.curselection()
        if selection and self.detections:
            index = selection[0]
            if 0 <= index < len(self.detections):
                self.target_object = self.detections[index]
                self.update_status(f"Target selected: {self.target_object['class_name']}")
    
    def on_servo_change(self, value):
        """Handle servo slider changes"""
        x_angle = self.x_servo_var.get()
        y_angle = self.y_servo_var.get()
        
        self.x_servo_label.config(text=f"{x_angle:.0f}°")
        self.y_servo_label.config(text=f"{y_angle:.0f}°")
        
        # Send command if connected
        if self.network_client.is_connected():
            task = asyncio.create_task(self.send_servo_async(x_angle, y_angle))
            # Keep reference to prevent garbage collection
            self._pending_tasks = getattr(self, '_pending_tasks', [])
            self._pending_tasks.append(task)
    
    def start_video_recording(self):
        """Start video recording when enemy detected (like Java version)"""
        if not self.video_recording.get():
            return
            
        if self.video_writer is None:
            timestamp = datetime.now().strftime("%d_%m_%y-%H_%M")
            filename = f"cat_tracking_{timestamp}.mp4"
            
            # Get frame dimensions
            if self.current_frame is not None:
                height, width = self.current_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(filename, fourcc, 24.0, (width, height))
                self.update_status(f"📹 Video-Aufnahme gestartet: {filename}")
                logger.info(f"Started video recording: {filename}")
    
    def stop_video_recording(self):
        """Stop video recording"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.update_status("📹 Video-Aufnahme beendet")
            logger.info("Stopped video recording")
    
    def record_frame(self):
        """Record current frame if recording is active"""
        if self.video_writer is not None and self.current_frame is not None:
            self.video_writer.write(self.current_frame)
    
    def fire_command(self):
        """Send fire command (like Java version)"""
        if self.fire_enabled.get() and self.target_object:
            self.update_status("💥 FIRE COMMAND EXECUTED!")
            logger.info("Fire command executed")
            
            # Ensure recording for fire events
            if self.video_recording.get():
                self.record_frame()
            
            # Send fire command to server if connected
            if self.network_client.is_connected():
                task = asyncio.create_task(self.network_client.send_fire_command())
                self._pending_tasks = getattr(self, '_pending_tasks', [])
                self._pending_tasks.append(task)
    
    def reset_servos(self):
        """Reset servos to center position (like Java version)"""
        self.x_servo_var.set(90)
        self.y_servo_var.set(90)
        self.update_status("🔄 Servos reset to center position", force_log=True)
        self.servo_center_reported = True  # Mark as reported
        
        # Send reset command to server
        if self.network_client.is_connected():
            task = asyncio.create_task(self.send_servo_async(90, 90))
            self._pending_tasks = getattr(self, '_pending_tasks', [])
            self._pending_tasks.append(task)
    
    def toggle_tracking(self):
        """Toggle YOLO tracking"""
        if self.tracking_enabled:
            self.tracking_enabled = False
            self.tracking_btn.config(text="🎯 Start Tracking")
            self.update_status("YOLO tracking stopped")
        else:
            if self.yolo_tracker.model is None:
                if self.yolo_tracker.initialize():
                    self.tracking_enabled = True
                    self.tracking_btn.config(text="⏸️ Stop Tracking")
                    self.update_status("YOLO tracking started")
                else:
                    self.update_status("❌ Failed to initialize YOLO model")
            else:
                self.tracking_enabled = True
                self.tracking_btn.config(text="⏸️ Stop Tracking")
                self.update_status("YOLO tracking started")
    
    def toggle_overlays(self):
        """Toggle detection overlays"""
        self.overlays_enabled = not self.overlays_enabled
        if self.overlays_enabled:
            self.toggle_overlays_btn.config(text="🔍 Hide Overlays")
            self.update_status("Detection overlays enabled")
        else:
            self.toggle_overlays_btn.config(text="🔍 Show Overlays")
            self.update_status("Detection overlays disabled")
    
    def add_friend(self):
        """Add selected class as friend"""
        selection = self.friend_combo.get()
        if selection:
            try:
                # Parse class ID from selection (format: "id: name")
                class_id = int(selection.split(':')[0])
                class_name = selection.split(': ')[1]
                
                if class_id not in self.friend_classes:
                    self.friend_classes.add(class_id)
                    # Remove from enemies if present
                    if class_id in self.enemy_classes:
                        self.enemy_classes.remove(class_id)
                    self.update_classification_display()
                    self.update_status(f"👥 {class_name} als Freund hinzugefügt", force_log=True)
                else:
                    self.update_status(f"👥 {class_name} ist bereits Freund")
            except (ValueError, IndexError):
                self.update_status("❌ Ungültige Auswahl")
    
    def remove_friend(self):
        """Remove selected class from friends"""
        selection = self.friend_combo.get()
        if selection:
            try:
                # Parse class ID from selection
                class_id = int(selection.split(':')[0])
                class_name = selection.split(': ')[1]
                
                if class_id in self.friend_classes:
                    self.friend_classes.remove(class_id)
                    self.update_classification_display()
                    self.update_status(f"👥 {class_name} als Freund entfernt", force_log=True)
                else:
                    self.update_status(f"👥 {class_name} ist kein Freund")
            except (ValueError, IndexError):
                self.update_status("❌ Ungültige Auswahl")
    
    def add_enemy(self):
        """Add selected class as enemy"""
        selection = self.enemy_combo.get()
        if selection:
            try:
                # Parse class ID from selection
                class_id = int(selection.split(':')[0])
                class_name = selection.split(': ')[1]
                
                if class_id not in self.enemy_classes:
                    self.enemy_classes.add(class_id)
                    # Remove from friends if present
                    if class_id in self.friend_classes:
                        self.friend_classes.remove(class_id)
                    self.update_classification_display()
                    self.update_status(f"👤 {class_name} als Feind hinzugefügt", force_log=True)
                else:
                    self.update_status(f"👤 {class_name} ist bereits Feind")
            except (ValueError, IndexError):
                self.update_status("❌ Ungültige Auswahl")
    
    def remove_enemy(self):
        """Remove selected class from enemies"""
        selection = self.enemy_combo.get()
        if selection:
            try:
                # Parse class ID from selection
                class_id = int(selection.split(':')[0])
                class_name = selection.split(': ')[1]
                
                if class_id in self.enemy_classes:
                    self.enemy_classes.remove(class_id)
                    self.update_classification_display()
                    self.update_status(f"👤 {class_name} als Feind entfernt", force_log=True)
                else:
                    self.update_status(f"👤 {class_name} ist kein Feind")
            except (ValueError, IndexError):
                self.update_status("❌ Ungültige Auswahl")
    
    def update_classification_display(self):
        """Update classification display text"""
        # Try to update if the display exists
        try:
            self.classification_display.delete(1.0, tk.END)
            
            # Show friends
            friend_names = [COCO_CLASSES[i] for i in sorted(self.friend_classes) if i < len(COCO_CLASSES)]
            enemy_names = [COCO_CLASSES[i] for i in sorted(self.enemy_classes) if i < len(COCO_CLASSES)]
            
            display_text = f"👥 FREUNDE ({len(friend_names)}):\n"
            display_text += ", ".join(friend_names) if friend_names else "Keine"
            display_text += f"\n\n👤 FEINDE ({len(enemy_names)}):\n"  
            display_text += ", ".join(enemy_names) if enemy_names else "Keine"
            
            self.classification_display.insert(1.0, display_text)
        except AttributeError:
            pass  # Display not initialized yet
    
    def on_friend_selection(self, event=None):
        """Handle friend class selection changes"""
        self.update_friend_enemy_classes()
        self.update_status("Friend classes updated")
    
    def on_enemy_selection(self, event=None):
        """Handle enemy class selection changes"""
        self.update_friend_enemy_classes()
        self.update_status("Enemy classes updated")
    
    def update_friend_enemy_classes(self):
        """Update friend and enemy class sets from dropdowns"""
        # Get selected friend classes
        friend_selection = self.friend_listbox.curselection()
        self.friend_classes = set()
        for i in friend_selection:
            class_name = self.friend_listbox.get(i)
            # Find class ID
            for class_id, name in enumerate(COCO_CLASSES):
                if name == class_name:
                    self.friend_classes.add(class_id)
                    break
        
        # Get selected enemy classes
        enemy_selection = self.enemy_listbox.curselection()
        self.enemy_classes = set()
        for i in enemy_selection:
            class_name = self.enemy_listbox.get(i)
            # Find class ID
            for class_id, name in enumerate(COCO_CLASSES):
                if name == class_name:
                    self.enemy_classes.add(class_id)
                    break
        
        logger.info(f"Updated classes - Friends: {self.friend_classes}, Enemies: {self.enemy_classes}")
    
    def on_close(self):
        """Clean shutdown"""
        # Stop recording if active
        if self.video_writer is not None:
            self.stop_video_recording()
        
        # Stop local camera
        if hasattr(self, 'local_camera'):
            self.local_camera.stop_capture()
        
        # Disconnect from server
        if hasattr(self, 'network_client'):
            try:
                asyncio.run(self.network_client.disconnect())
            except:
                pass
        
        # Close window
        self.root.destroy()
    
    def run(self):
        """Start the GUI application"""
        try:
            logger.info("🚀 Starting Enhanced Cat Tracking GUI")
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted")
        finally:
            # Cleanup
            if self.video_writer:
                self.video_writer.release()
            if self.network_client and self.network_client.is_connected():
                try:
                    # Use sync disconnect instead of async
                    self.network_client.stop_video_stream()
                    if self.network_client.websocket:
                        # Close websocket synchronously
                        self.network_client.connected = False
                except Exception as e:
                    logger.warning(f"Cleanup error: {e}")

def main():
    """Main entry point"""
    print("🐱 Cat Tracking System - Python CUDA Edition (Enhanced)")
    print("=" * 60)
    print("🚀 All Java features + Ultra-high performance YOLO with CUDA")
    print("📈 Expected performance: 30-60 FPS (vs 5-10 FPS Java)")
    print("⚔️ Combat ready with friend/enemy classification")
    print()
    
    app = CatTrackingGUI()
    app.run()

if __name__ == '__main__':
    main()
