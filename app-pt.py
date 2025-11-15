"""
Facial Emotion Detection using YOLO (You Only Look Once)
=========================================================
Real-time emotion detection application for Raspberry Pi using YOLOv11 nano model.
Detects faces and classifies emotions: Happy, Sad, Angry, Excited, Fear, Disgust, 
Serious, Thinking, Worried, and Neutral.

Supports both Raspberry Pi Camera Module and USB webcams with automatic color format detection.
"""

from ultralytics import YOLO
import cv2
from picamera2 import Picamera2
import torch
import os

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load pre-trained YOLOv11 Nano model fine-tuned for emotion detection
# Uses relative path so it works from any directory
model_path = os.path.join(script_dir, 'yolo-trained-models', 'emotionsbest.pt')

# Verify model exists before loading
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    exit(1)

# The model returns bounding boxes and emotion class predictions
model = YOLO(model_path)

# Print all class names for verification
# Output: {0: 'Angry', 1: 'Disgust', 2: 'Excited', 3: 'Fear', 4: 'Happy', 5: 'Sad', 6: 'Serious', 7: 'Thinking', 8: 'Worried', 9: 'neutral'}
print(model.names)

# ============================================================================
# EMOTION CLASSIFICATION AND COLOR MAPPING
# ============================================================================

# Get class names from model (emotion labels)
CLASS_MAP = model.names

# Define BGR color palette for each emotion (OpenCV uses BGR, not RGB)
# Colors are chosen to be visually distinct and intuitive
STANDARD_COLORS = {
    "Happy": (0, 255, 0),           # Green
    "Sad": (255, 191, 0),           # Cyan-Blue
    "Angry": (0, 0, 255),           # Red
    "Excited": (0, 165, 255),       # Orange
    "Fear": (0, 0, 128),            # Dark Red
    "Disgust": (128, 0, 128),       # Purple
    "Serious": (0, 128, 0),         # Dark Green
    "Thinking": (0, 128, 0),        # Dark Green
    "Worried": (0, 128, 0),         # Dark Green
    "neutral": (192, 192, 192)      # Gray
}

# Map emotion class names to their corresponding colors
EMOTION_COLORS = {name: STANDARD_COLORS.get(name, (255, 255, 255)) for name in CLASS_MAP.values()}

# ============================================================================
# CAMERA INITIALIZATION AND SELECTION
# ============================================================================

# Get list of all available cameras (Raspberry Pi Camera Module and USB cameras)
camera_list = Picamera2.global_camera_info()

# Check if any cameras are detected
if len(camera_list) == 0:
    print('No camera detected.')
    exit()
elif len(camera_list) == 1:
    # Only one camera available, use it automatically
    camera_instance = 0
else:
    # Multiple cameras detected, prompt user to select one
    print('\nThe following cameras were detected:')
    camera_num = 1
    for camera in camera_list:
        # Map camera model ID to human-readable name
        if camera['Model'] == 'imx708':
            camera_model = 'Raspberry Pi Camera Module'
        else:
            camera_model = camera['Model']
        print(camera_num, camera_model)
        camera_num = camera_num + 1
    
    # Get user input for camera selection
    try:
        camera_instance = int(input('Enter the number for the camera that you wish to use: '))
    except:
        print('Invalid camera number.')
        exit()
    
    # Validate camera selection
    if (camera_instance > len(camera_list)):
        print('Invalid camera number.')
        exit()
    else:
        camera_instance = camera_instance - 1

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================

# Detect if USB camera is being used (requires color format conversion)
# USB cameras typically use BGR format, while Raspberry Pi cameras use RGB
if 'usb' in camera_list[camera_instance]['Id']:
    webcam_color_shift = True  # Need to convert BGR to RGB
else:
    webcam_color_shift = False

# Initialize camera with selected instance
picam2 = Picamera2(camera_instance)

# Configure camera resolution: High-res for better emotion detection accuracy
picam2.preview_configuration.main.size = (3280, 2464)  # Full resolution

# Set color format based on camera type
if webcam_color_shift:
    picam2.preview_configuration.main.format = "BGR888"  # USB camera format
else:
    picam2.preview_configuration.main.format = "RGB888"  # Raspberry Pi camera format

# Align configuration to camera requirements
picam2.preview_configuration.align()

# Apply configuration and start camera stream
picam2.configure("preview")
picam2.start()

# ============================================================================
# MAIN DETECTION LOOP
# ============================================================================

frame_count = 0
N = 2  # Process inference every 2nd frame (skip frames to improve performance)

while True:
    frame_count += 1
    
    # Capture frame from camera
    frame = picam2.capture_array()

    # Convert color format if using USB camera
    if webcam_color_shift:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run emotion detection on every Nth frame (skip for performance)
    if frame_count % N == 0:
        # Forward pass through YOLO model
        results = model(frame)

        # Prepare frame for drawing annotations
        annotated_frame = frame.copy()
        
        # Draw bounding boxes and emotion labels manually for better control
        if hasattr(results[0], "boxes") and results[0].boxes is not None:
            for box in results[0].boxes:
                # Extract bounding box coordinates (pixel positions)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get emotion class ID and convert to label
                cls_id = int(box.cls[0])
                emotion = CLASS_MAP.get(cls_id, "neutral")
                
                # Get color for this emotion
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                
                # Convert confidence score from decimal (0-1) to percentage (0-100)
                conf_num = int(float(box.conf[0]) * 100)
                
                # Create label text with emotion and confidence
                label = f"{emotion} {conf_num}%"

                # ====== DRAW LABEL BACKGROUND ======
                # Calculate text size for proper background sizing
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                # Draw colored rectangle as label background (above the bounding box)
                cv2.rectangle(annotated_frame,
                    (x1, y1 - text_h - 15),        # Top-left corner (above box)
                    (x2, y1),                       # Bottom-right corner
                    color,                          # Emotion-specific color
                    -1)                             # -1 fills the rectangle
                
                # ====== DRAW BOUNDING BOX ======
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # ====== DRAW TEXT LABEL ======
                cv2.putText(annotated_frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # ====== DISPLAY FPS (Frames Per Second) ======
        # Calculate FPS from model inference time
        inference_time = results[0].speed['inference']  # Time in milliseconds
        fps = 1000 / inference_time  # Convert to FPS
        text = f'FPS: {fps:.1f}'
        
        # Position FPS display in top-right corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = annotated_frame.shape[1] - text_size[0] - 10  # Right-aligned with 10px margin
        text_y = text_size[1] + 10                             # Top-aligned with 10px margin
        
        # Draw FPS text in white
        cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # ====== DISPLAY THE FRAME ======
        title = 'Emotion Detection'
        cv2.imshow(title, annotated_frame)

        # ====== EXIT CONDITION ======
        # Press 'q' to quit the program
        if cv2.waitKey(10) == ord("q"):
            break

# ============================================================================
# CLEANUP
# ============================================================================

# Close all OpenCV windows on exit
cv2.destroyAllWindows()
