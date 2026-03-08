
Vehicle tracking readme · MD
Copy

# Vehicle Tracking with Optical Flow and DeepSORT

A real-time vehicle tracking system that combines YOLOv8 object detection, DeepSORT tracking, and optical flow for video stabilization and motion analysis.

## Overview

This project implements a robust vehicle tracking pipeline that processes racing car videos to:
- Detect vehicles using YOLOv8
- Track individual vehicles across frames using DeepSORT
- Analyze motion patterns using optical flow
- Provide frame-by-frame tracking coordinates
- Stabilize video using affine transformations

## Features

### 1. **Multi-Algorithm Tracking**
- YOLOv8m model for accurate vehicle detection
- DeepSORT tracker with configurable age parameter (max_age=20)
- Lucas-Kanade optical flow for motion estimation
- Good Features to Track for feature point selection

### 2. **Video Processing**
- Top-half frame cropping to eliminate false detections from car interiors
- Real-time optical flow vector visualization
- Affine transformation-based video stabilization
- Frame-by-frame coordinate logging

### 3. **Performance Optimization**
- Processes only the upper 50% of frames (reduces computational load)
- Efficient feature tracking with configurable parameters
- Real-time inference and tracking display

## System Architecture

```
Video Input
    ↓
Frame Extraction → Top Half Cropping
    ↓
Optical Flow Calculation
    ↓
Affine Transformation (Stabilization)
    ↓
YOLOv8 Detection
    ↓
DeepSORT Tracking
    ↓
Visualization & Coordinate Logging
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for real-time processing)

### Required Libraries

```bash
pip install ultralytics
pip install deep-sort-realtime
pip install opencv-python
pip install numpy
pip install matplotlib
```

## Usage

### Basic Implementation

```python
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize models
model = YOLO("yolov8m.pt")
tracker = DeepSort(max_age=20)

# Load video
cap = cv2.VideoCapture("carrace.mp4")

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Crop top half
    height, width, _ = frame.shape
    top_half = frame[:height // 2, :]
    
    # Run detection and tracking
    results = model(top_half)
    # ... (tracking pipeline)
```

### Running the Notebook

1. Open `mainreal.ipynb` in Jupyter Notebook or JupyterLab
2. Ensure `carrace.mp4` or `carrace2.mp4` is in the same directory
3. Run cells sequentially to execute different tracking approaches

## Code Structure

### Implementation 1: Basic Optical Flow + YOLOv8 + DeepSORT

**File Location:** Cell 1 in `mainreal.ipynb`

**Features:**
- Farneback optical flow method
- Motion vector visualization with red arrows
- Top-half frame processing
- Real-time motion tracking

**Key Parameters:**
```python
cv2.calcOpticalFlowFarneback(
    prev_gray, curr_gray, None, 
    pyr_scale=0.5, 
    levels=3, 
    winsize=15, 
    iterations=3, 
    poly_n=5, 
    poly_sigma=1.2, 
    flags=0
)
```

### Implementation 2: YOLO Detection + DeepSORT Tracking

**File Location:** Cell 2 in `mainreal.ipynb`

**Features:**
- Pure YOLO + DeepSORT pipeline
- Configurable tracker age (max_age=20)
- Bounding box visualization
- Track ID assignment and display

**Detection Parameters:**
```python
DeepSort(max_age=20)  # Tracks persist for 20 frames after last detection
```

### Implementation 3: Advanced Optical Flow with Stabilization

**File Location:** Cell 3 in `mainreal.ipynb`

**Features:**
- Lucas-Kanade optical flow
- Affine transformation for video stabilization
- Feature point tracking
- Coordinate logging for each tracked object

**Optical Flow Parameters:**
```python
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)
```

## Technical Details

### YOLOv8 Model
- **Model Version:** YOLOv8m (medium)
- **Input Resolution:** 192×640
- **Average Inference Time:** 12-22ms per frame
- **Classes Detected:** cars, trucks, buses, motorcycles, and other vehicles

### DeepSORT Tracker
- **Maximum Age:** 20 frames (configurable)
- **Tracking Method:** Kalman filtering + appearance features
- **ID Assignment:** Persistent across occlusions
- **Re-identification:** Enabled for lost tracks

### Optical Flow
- **Method 1:** Farneback (dense optical flow)
- **Method 2:** Lucas-Kanade (sparse optical flow)
- **Purpose:** Motion estimation and video stabilization
- **Visualization:** Red arrow vectors showing motion direction

## Performance Metrics

### Processing Speed
- **Average Preprocessing:** 1.0-1.8ms per frame
- **Average Inference:** 12-22ms per frame
- **Average Postprocessing:** 1.5-3.5ms per frame
- **Total Processing:** 15-27ms per frame (~37-66 FPS)

### Detection Accuracy
- Successfully detects 4-8 cars per frame in typical racing scenarios
- Maintains tracking through partial occlusions
- Handles scale variations and perspective changes

## Output Format

### Console Output
```
Tracked Object 6: x1=24, y1=1, x2=850, y2=235
Tracked Object 11: x1=419, y1=159, x2=443, y2=172
Tracked Object 14: x1=420, y1=175, x2=439, y2=187
```

**Coordinate System:**
- `(x1, y1)`: Top-left corner of bounding box
- `(x2, y2)`: Bottom-right corner of bounding box
- Coordinates relative to top-half of original frame

### Visual Output
- Green bounding boxes around tracked vehicles
- Track IDs displayed above each box
- Red motion vectors (in optical flow implementations)
- Real-time display window: "Processed (Top Half Only)"

## Troubleshooting

### Common Issues

**1. Video Not Loading**
```python
# Check video path
cap = cv2.VideoCapture("path/to/your/video.mp4")
if not cap.isOpened():
    print("Error: Cannot open video file")
```

**2. Low FPS / Slow Processing**
- Use GPU acceleration (ensure CUDA is properly installed)
- Reduce input video resolution
- Use YOLOv8n (nano) instead of YOLOv8m (medium)
- Decrease `maxCorners` in optical flow parameters

**3. Lost Tracks**
- Increase `max_age` parameter in DeepSORT
- Adjust detection confidence threshold
- Improve lighting conditions in source video

**4. False Detections**
- Increase YOLO confidence threshold
- Fine-tune cropping region to exclude unwanted areas
- Use class filtering to only track vehicles

## Customization

### Adjusting Detection Region

```python
# Change crop ratio (default is top 50%)
crop_ratio = 0.6  # Use top 60% of frame
top_region = frame[:int(height * crop_ratio), :]
```

### Modifying Tracking Parameters

```python
# Increase tracking persistence
tracker = DeepSort(max_age=30)  # Keep tracks for 30 frames

# Adjust optical flow sensitivity
feature_params = dict(
    maxCorners=150,      # More feature points
    qualityLevel=0.2,    # Lower quality threshold
    minDistance=5        # Closer point spacing
)
```

### Changing YOLO Model

```python
# For faster processing (less accurate)
model = YOLO("yolov8n.pt")  # Nano model

# For better accuracy (slower)
model = YOLO("yolov8x.pt")  # Extra-large model
```

## Future Enhancements

### Planned Features
- [ ] Multi-camera fusion
- [ ] Speed estimation from optical flow
- [ ] Trajectory prediction using Kalman filters
- [ ] Vehicle re-identification across camera views
- [ ] Export tracking data to CSV/JSON
- [ ] Real-time analytics dashboard
- [ ] Integration with lane detection

### Research Directions
- Deep learning-based optical flow (PWC-Net, RAFT)
- Transformer-based object tracking
- 3D reconstruction from motion
- Anomaly detection in racing scenarios

## Dependencies

```
opencv-python>=4.8.0
ultralytics>=8.0.0
deep-sort-realtime>=1.3.2
numpy>=1.24.0
matplotlib>=3.7.0
```

## License

This project uses:
- YOLOv8 (AGPL-3.0 License)
- DeepSORT (GPL-3.0 License)
- OpenCV (Apache 2.0 License)

## Acknowledgments

- **YOLOv8:** Ultralytics team for state-of-the-art object detection
- **DeepSORT:** Original paper by Wojke et al.
- **OpenCV:** Computer vision foundation library
- **Racing Footage:** Source video providers

## Contributing

Contributions are welcome! Areas for improvement:
- Algorithm optimization
- Additional tracking metrics
- Documentation enhancements
- Bug fixes and testing

## Contact

For questions, issues, or collaboration:
- Create an issue in the repository
- Submit pull requests for improvements
- Share your tracking results and use cases
