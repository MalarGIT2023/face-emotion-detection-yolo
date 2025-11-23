![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Required-red)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-v11-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange)
![PiCamera2](https://img.shields.io/badge/PiCamera2-libcamera%20API-lightgrey)
![RaspberryPi](https://img.shields.io/badge/Raspberry%20Pi-Compatible-green)
![Model](https://img.shields.io/badge/AI-Real--Time--Inference-purple)
![License](https://img.shields.io/badge/License-MIT-green)
[![Stars](https://img.shields.io/github/stars/MalarGIT2023/face-emotion-detection-yolo?style=social)](https://github.com/MalarGIT2023/face-emotion-detection-yolo)

# Facial Emotion Detection using YOLOv11

**Mission Tomorrow Career Exploration Event hosted by Chamber RVA**  
*Presented to 11,000+ eighth graders in Richmond*  
*Volunteered for IEEE Region3 Richmond*

---

Real-time facial emotion detection application for **Raspberry Pi** using YOLOv11 Nano model. Detects faces and classifies emotions with high accuracy on edge devices.

## About This Project

This project was created for the **Mission Tomorrow Career Exploration Event** — a career awareness initiative by Chamber RVA for eighth-grade students in Richmond. The project demonstrates real-world artificial intelligence and machine learning applications in emotion recognition technology.

### Learning Objectives

Through this project, students explore:
- **Artificial Intelligence (AI)**: How machines learn to recognize patterns
- **Machine Learning (ML)**: Training models on real data
- **Computer Vision**: Analyzing images and detecting objects
- **Edge Computing**: Running AI on resource-constrained devices (Raspberry Pi)
- **Data Science Workflow**: Dataset collection → Model training → Deployment
- **Career Paths**: AI/ML engineer, Data scientist, Computer vision specialist

### Why Emotion Detection?

Emotion recognition has real-world applications in:
- **Mental health monitoring** — Detecting emotional distress
- **Human-computer interaction** — Responsive interfaces
- **Market research** — Analyzing customer reactions
- **Accessibility** — Assisting non-verbal communication
- **Entertainment** — Adaptive gaming and content

## Project Overview

This is the **Inference & Deployment** component of the emotion detection system. It uses a pre-trained YOLOv11 model to perform real-time emotion classification on Raspberry Pi and other edge devices.

### Complete Workflow

```
Step 1: Dataset Management          Step 2: Model Training          Step 3: Inference & Deploy
┌─────────────────────────┐         ┌──────────────────────┐       ┌───────────────────────────┐
│ roboflow-dataset-manager│   →     │ yolo-model-training  │   →   │face-emotion-detection-yolo│
│  (SEPARATE REPO)        │         │  (SEPARATE REPO)     │       │   (THIS PROJECT)          │
│ • Find datasets         │         │ • Train YOLOv11      │       │                           │
│ • Download from         │         │ • Fine-tune weights  │       │ • Real-time detection     │
│   Roboflow Universe     │         │ • Save best.pt       │       │ • Demo scripts            │
└─────────────────────────┘         └──────────────────────┘       │ • Easy deployment         │
                                                                   └───────────────────────────┘
```

## Features

- **Real-time emotion classification**: 10 emotion categories (Angry, Disgust, Excited, Fear, Happy, Sad, Serious, Thinking, Worried, Neutral)
- **Optimized for Raspberry Pi**: Uses YOLOv11 Nano model for fast inference on low-power hardware
- **Multi-camera support**: Works with Raspberry Pi Camera Module and USB webcams
- **Automatic color format handling**: Detects and converts camera color formats (BGR/RGB)
- **High-resolution processing**: Processes frames at 3280×2464 resolution for accuracy
- **Real-time FPS display**: Shows inference performance metrics
- **Color-coded emotions**: Each emotion has a distinct color for easy visual identification
- **Easy demo script**: Launch with a single shell script command

## Supported Emotions

| Emotion | Color | RGB Code |
|---------|-------|----------|
| Happy | Green | (0, 255, 0) |
| Sad | Cyan-Blue | (255, 191, 0) |
| Angry | Red | (0, 0, 255) |
| Excited | Orange | (0, 165, 255) |
| Fear | Dark Red | (0, 0, 128) |
| Disgust | Purple | (128, 0, 128) |
| Serious | Dark Green | (0, 128, 0) |
| Thinking | Dark Green | (0, 128, 0) |
| Worried | Dark Green | (0, 128, 0) |
| Neutral | Gray | (192, 192, 192) |

## Requirements

### Hardware
- **Raspberry Pi 5** (recommended) or Raspberry Pi 4 with 2GB+ RAM
- **Camera**: Raspberry Pi Camera Module (IMX708) or USB Webcam
- **Display**: HDMI monitor or headless mode via SSH

### Software
- Python 3.8+
- Raspberry Pi OS (Bullseye or later)
- CUDA/GPU support optional (for faster inference)

### Python Dependencies
- `ultralytics` — YOLOv11 implementation
- `opencv-python` — Computer vision library
- `picamera2` — Raspberry Pi camera interface
- `torch` — Deep learning framework
- `torchvision` — PyTorch vision utilities

## Installation

### Prerequisites: Trained Model Required

This project is the **deployment/inference stage**. It requires a trained emotion detection model from a separate training repository. You must complete these steps first:

**1. Prepare Dataset (Separate Repository)**
Clone and run the dataset manager from its own repository:
```bash
git clone https://github.com/MalarGIT2023/roboflow-dataset-manager.git
cd roboflow-dataset-manager
python dataset-download.py
# See the repository's README.md for detailed instructions
```

**2. Train Model (Separate Repository)**
Clone and run the model training from its own repository:
```bash
git clone https://github.com/MalarGIT2023/yolo-model-training.git
cd yolo-model-training
python model-training.py
# Output: runs/detect/train/weights/best.pt
# See the repository's README.md for training details
```

**3. Copy Trained Model**
```bash
cp ../yolo-model-training/runs/detect/train/weights/best.pt \
   ./yolo-trained-models/emotionsbest.pt
```

### Installation Steps

#### Step 1: Clone the Repository
```bash
git clone https://github.com/MalarGIT2023/face-emotion-detection-yolo.git
cd face-emotion-detection-yolo
```

#### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv face-emotion-yolo-venv
source face-emotion-yolo-venv/bin/activate  # On Windows: face-emotion-yolo-venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Verify Model File
Check that the trained model exists in the correct location:
```bash
ls yolo-trained-models/emotionsbest.pt
# If this fails, follow the Prerequisites section above
```

### Step 5: Run the Application

**Option A: Using the demo script (Recommended)**
```bash
chmod +x demo/face-emotion-yolo.sh
./demo/face-emotion-yolo.sh
```

**Option B: Direct Python execution**
```bash
# Ensure virtual environment is activated
source face-emotion-yolo-venv/bin/activate
python app-pt.py
```

## Usage

1. **Using the demo script** — Run `./demo/face-emotion-yolo.sh`
2. **Select camera** — If multiple cameras are detected, choose which one to use
3. **View live detection** — Emotion labels with confidence scores appear in real-time
4. **Monitor performance** — Frame Per Second (FPS) is displayed in the top-right corner
5. **Exit** — Press **'q'** to quit the application

## Project Structure

**This Repository** (face-emotion-detection-yolo - Inference & Deployment):
```
face-emotion-detection-yolo/
├── app-pt.py
├── requirements.txt
├── README.md (this file)
├── demo/
│   └── face-emotion-yolo.sh
└── yolo-trained-models/
    └── emotionsbest.pt               ← Trained model (copied from training repo)
```

**Related Separate Repositories**:

1. **roboflow-dataset-manager** — Dataset management and downloading
   - Repository: Separate GitHub repository
   - Purpose: Download and prepare emotion datasets
   - See its own README for structure and details

2. **yolo-model-training** — Model training
   - Repository: Separate GitHub repository
   - Purpose: Train YOLOv11 models on emotion datasets
   - Output: `runs/detect/train/weights/best.pt`
   - See its own README for structure and details

## Technical Details

### Model Architecture
- **Base Model**: YOLOv11 Nano (lightweight and fast)
- **Fine-tuned Models**: 
  - `yolo11n.pt` — Nano model (baseline, fastest)
  - `yolo11l.pt` — Large model (more accurate, slower)
  - `emotionsbest.pt` — Custom trained Nano model
- **Input Resolution**: 3280×2464 (full Raspberry Pi Camera resolution)
- **Classes**: 10 emotions
- **Framework**: PyTorch via Ultralytics

### Processing Pipeline
1. **Camera Capture** → Full-resolution frame from Picamera2
2. **Color Conversion** → Automatic BGR/RGB format detection
3. **Frame Skipping** → Process every 2nd frame (configurable via `N`)
4. **YOLO Inference** → Detect faces and classify emotions
5. **Annotation** → Draw boxes, labels, and confidence scores
6. **Display** → Show annotated frame with FPS counter

### Performance Optimization
- **Frame Skipping**: Processes every 2nd frame to reduce computational load
- **Nano Model**: Smaller model size for faster inference
- **CPU-based**: Works without GPU (GPU support available for faster processing)
- **Resolution Adaptive**: Handles different camera resolutions automatically

### Camera Support
- **Raspberry Pi Camera Module** (IMX708) — Uses RGB888 format
- **USB Webcam** — Detects USB cameras and converts from BGR to RGB automatically
- **Multiple Cameras** — Interactive selection if multiple cameras connected

## Configuration

### Adjust Frame Processing Speed
Edit the `N` variable in `app-pt.py`:
```python
N = 2  # Process every 2nd frame (higher = faster but less frequent inference)
```

### Camera Resolution
Change the resolution in `app-pt.py`:
```python
picam2.preview_configuration.main.size = (3280, 2464)  # Current: Full resolution
# Or use lower resolution for faster processing:
# picam2.preview_configuration.main.size = (1920, 1080)  # Full HD
```

### Emotion Color Codes
Modify the `STANDARD_COLORS` dictionary in `app-pt.py` to customize emotion colors.

## Troubleshooting

### No Camera Detected
```bash
# Check camera connection
libcamera-hello --list-cameras

# Verify camera is enabled in raspi-config
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
```

### Low FPS / Slow Performance
- Lower resolution: Change `main.size` to (1920, 1080) or smaller
- Increase frame skipping: Set `N = 3` or higher
- Use GPU acceleration if available: Install `torch` with CUDA support

### Import Errors
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade

# Or specific package
pip install ultralytics --upgrade
```

### Camera Format Issues
The script automatically detects USB vs Raspberry Pi cameras. If colors appear inverted:
- Check the `webcam_color_shift` variable logic
- Manually set to `True` for USB or `False` for Raspberry Pi cameras

### Out of Memory (OOM) Errors
- Lower resolution or frame size
- Increase frame skipping (`N`)
- Close other applications running on Pi

## Related Projects

This project is part of the IEEE Mission Tomorrow emotion detection application. It's designed to work with two other **separate repositories**:

### 1. **roboflow-dataset-manager** (Separate Repository)
Manages emotion detection datasets from Roboflow Universe.

**Repository**: Separate GitHub repository  
**Purpose**:
- Download emotion detection datasets
- Access Roboflow Universe
- Prepare data in YOLOv11 format

**Setup**:
```bash
git clone https://github.com/MalarGIT2023/roboflow-dataset-manager.git
cd roboflow-dataset-manager
python dataset-download.py
```

**See**: That repository's README.md for detailed instructions.

### 2. **yolo-model-training** (Separate Repository)
Trains YOLOv11 models on emotion datasets.

**Repository**: Separate GitHub repository  
**Purpose**:
- Fine-tune YOLOv11 on emotion data
- Generate trained weights
- Export best model

**Setup**:
```bash
git clone https://github.com/MalarGIT2023/yolo-model-training.git
cd yolo-model-training
python model-training.py
```

**See**: That repository's README.md for detailed instructions.

### 3. **This Project: face-emotion-detection-yolo**
Real-time inference on Raspberry Pi (deployment stage).

### Complete Workflow

```
roboflow-dataset-manager (separate repo)
           ↓
yolo-model-training (separate repo)
           ↓
face-emotion-detection-yolo (this repo - inference)
```

### Complete Workflow Example

```bash
# Step 1: Prepare dataset (in separate repository)
git clone https://github.com/MalarGIT2023/roboflow-dataset-manager.git
cd roboflow-dataset-manager
python dataset-download.py
cd ..

# Step 2: Train model (in separate repository)
git clone https://github.com/MalarGIT2023/yolo-model-training.git
cd yolo-model-training
python model-training.py
# Output: runs/detect/train/weights/best.pt
cd ..

# Step 3: Deploy for inference (this repository)
git clone https://github.com/MalarGIT2023/face-emotion-detection-yolo.git
cd face-emotion-detection-yolo
cp ../yolo-model-training/runs/detect/train/weights/best.pt \
   ./yolo-trained-models/emotionsbest.pt

# Step 4: Run the application
./demo/face-emotion-yolo.sh
```

## Performance Benchmarks

Typical performance on **Raspberry Pi 5** with YOLOv11 Nano:
- **Resolution**: 3280×2464
- **Frame Rate**: ~3-5 FPS (depending on inference load)
- **Latency**: ~200-300ms per frame
- **Accuracy**: ~85-90% on standard emotion datasets
- **Model Size**: ~6.5 MB (Nano model)

## Advanced Usage

### Headless Mode (SSH)
```bash
# Display results via SSH
DISPLAY=:0 python app-pt.py

# Or save results to file for later analysis
python app-pt.py > emotion_detection.log 2>&1 &
```

### Recording Detections
Modify `app-pt.py` to save frames with annotations:
```python
# Add VideoWriter after frame annotation
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
out.write(annotated_frame)
```

### Custom Model Fine-tuning
Use transfer learning with YOLOv11 models:
```bash
cd model-training
python model-training.py --model yolo11n.pt --data custom_dataset.yaml
```

### Using the Demo Script
The provided `face-emotion-yolo.sh` script:
- Automatically detects the virtual environment
- Activates the environment
- Runs the Python application with relative paths
- Can be run from any directory

```bash
# Make it executable
chmod +x demo/face-emotion-yolo.sh

# Run it
./demo/face-emotion-yolo.sh
```

### Pre-requisite: Trained Model
Before running the demo, ensure `emotionsbest.pt` exists in `yolo-trained-models/`. See the **Model Training** section for instructions.

## Notes

- Color format is automatically detected (BGR for USB, RGB for Raspberry Pi cameras)
- Confidence scores indicate model certainty (higher = more confident)
- Frame skipping improves speed but reduces detection frequency
- GPU acceleration can improve performance up to 10x (if CUDA-capable GPU available)
- The model is optimized for frontal face detection

## Support

For issues or questions:
1. Check the **Troubleshooting** section
2. Review terminal output for error messages
3. Verify camera connection and Raspberry Pi setup
4. Check model file exists in `yolo-trained-models/`

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) file for full details.

**MIT License Summary**: You are free to use, modify, and distribute this software for any purpose, provided you include the original license and copyright notice.

## Credits & Acknowledgments

**Created for**: IEEE Mission Tomorrow Career Exploration Event  
**Event**: Hosted by Chamber RVA for 11,000+ eighth graders in Richmond  
**Presented by**: IEEE Region 3 Richmond

**External Dependencies**:
- **Ultralytics YOLOv11** — Object detection framework
- **OpenCV** — Computer vision library
- **PyTorch** — Deep learning framework
- **Roboflow** — Dataset management and distribution
- **Raspberry Pi Foundation** — Raspberry Pi Camera interface libraries

**Dataset Sources**:
- Emotion datasets provided through Roboflow Universe

**Special Thanks**:
- IEEE Region 3 Richmond for volunteering
- Chamber RVA for organizing Mission Tomorrow
- All educators supporting STEM education

---

**Last Updated**: November 2025  
**Version**: 1.0  
**Optimized for**: Raspberry Pi 5 with YOLOv11 Nano  
**Component**: Inference & Deployment  
**Related Projects** (separate repositories): 
- roboflow-dataset-manager (Dataset management)
- yolo-model-training (Model training)

