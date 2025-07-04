# Face Mask Detector

Real-time face mask detection using deep learning and computer vision.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train.py
   ```

3. Run detection:
   ```bash
   python detect.py
   ```

## Files

- `train.py` - Train the mask detection model
- `detect.py` - Real-time mask detection
- `requirements.txt` - Dependencies
- `dataset/` - Training images (with_mask, without_mask)
- `architecture.txt` - Face detection model architecture
- `weights.caffemodel` - Pre-trained face detection weights

## Controls

- Press 's' to stop detection

## Requirements

- Python 3.7+
- Webcam
