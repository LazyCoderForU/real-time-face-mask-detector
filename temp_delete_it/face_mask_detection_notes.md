# Face Mask Detection Project - Learning Requirements

## Core Programming
1. **Python fundamentals**
   - Variables, loops, conditionals
   - Functions and error handling
   - Working with libraries/imports

## Computer Vision with OpenCV
1. **Basic image concepts**
   - Pixel representation and color spaces (BGR vs RGB)
   - Image dimensions and shapes

2. **Frame manipulation**
   - Reading video frames (`vs.read()`)
   - Resizing images (`imutils.resize()`)
   - Region extraction (face cropping)

3. **Drawing functions**
   - Rectangle drawing (`cv2.rectangle()`)
   - Text overlay (`cv2.putText()`)
   - Color formatting in OpenCV (BGR tuple format)

4. **Video handling**
   - Capturing from webcam
   - Continuous frame processing
   - Displaying frames (`cv2.imshow()`)
   - Keyboard input handling (`cv2.waitKey()`)

5. **DNN module**
   - Loading pre-trained models (`cv2.dnn.readNet()`)
   - Converting images to blobs (`cv2.dnn.blobFromImage()`)
   - Understanding model input preprocessing

## Deep Learning
1. **Keras fundamentals**
   - Model loading (`load_model()`)
   - Making predictions (`model.predict()`)
   - Batch processing

2. **Transfer learning concepts**
   - MobileNetV2 architecture basics
   - Image preprocessing (`preprocess_input()`)

3. **Classification concepts**
   - Confidence scores
   - Binary classification (mask vs. no mask)
   - Prediction interpretation

## Data Handling
1. **NumPy operations**
   - Array creation and manipulation
   - Data type conversions (`astype()`)
   - Array indexing and slicing

2. **Image preprocessing**
   - Converting to arrays (`img_to_array()`)
   - Normalization techniques
   - Reshaping for neural network input

## Project-Specific Knowledge
1. **Face detection workflow**
   - Understanding the model's output format
   - Confidence thresholding
   - Bounding box extraction

2. **Real-time processing techniques**
   - Efficient frame processing
   - Performance considerations
   - Memory management

3. **Model files**
   - Understanding what's in architecture.txt
   - Understanding what's in weights.caffemodel
   - Understanding the mask_detector.h5 model

This breakdown covers all technical aspects needed to fully understand and modify the face mask detection system.

---

# Real-Time Face Mask Detection: Practical Notes

## 1. Setup & Installation

### Theory
- This project uses OpenCV, Keras, TensorFlow, NumPy, and imutils
- Two neural networks work together: one for face detection, one for mask classification

### Practical Setup
```bash
# Create a virtual environment
python -m venv mask_env
source mask_env/bin/activate  # On Windows: mask_env\Scripts\activate

# Install required packages
pip install opencv-python tensorflow keras imutils numpy

# Download required model files
# You'll need:
# - architecture.txt (face detection prototxt)
# - weights.caffemodel (face detection weights)
# - mask_detector.h5 (trained mask detection model)
```

## 2. Understanding the Main Loop (Lines 130-165)

### Theory
- The main loop continuously captures video frames and processes them
- For each detected face, it classifies whether a mask is being worn
- Results are visually displayed with colored rectangles and confidence percentages
- Loop continues until the user presses 's' to stop

### Practical Exercise
```python
# Test your understanding by adding a counter for faces detected
face_count = 0

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (Array_locs, Array_preds) = detect_and_predict_mask(frame, face_nn, mask_nn)
    
    # Count faces in this frame
    face_count = len(Array_locs)
    
    # Add counter text to frame
    cv2.putText(frame, f"Faces: {face_count}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Rest of loop remains the same...
```

## 3. Visualization Components (Lines 130-153)

### Theory
- `zip(Array_locs, Array_preds)` pairs each face location with its prediction
- The prediction contains two values: confidence of mask and no-mask
- Green rectangles (0,255,0) indicate masks, red rectangles (0,0,255) indicate no masks
- OpenCV uses BGR color format (not RGB)
- Format strings create percentage labels showing confidence

### Practical Experience
```python
# Try different colors and text formats
# Original:
if label == "Mask":
    color = (0, 255, 0)  # Green in BGR
else:
    color = (0, 0, 255)  # Red in BGR

# Modified with additional orange warning for low confidence:
if mask > withoutMask:
    if mask > 0.8:  # High confidence
        color = (0, 255, 0)  # Green
    else:  # Low confidence
        color = (0, 165, 255)  # Orange
    label = "Mask"
else:
    label = "No Mask"
    color = (0, 0, 255)  # Red
```

## 4. Drawing on Frames (Lines 146-153)

### Theory
- `cv2.putText()` parameters explained:
  - frame: image to draw on
  - label: text to display
  - (startX, startY-10): position (10px above face box)
  - font face and scale: font type and size
  - color: BGR tuple
  - thickness: line thickness
- `cv2.rectangle()` draws bounding boxes around faces

### Practical Exercise
```python
# Experiment with different text positions and styles
# Add a background to text for better visibility
text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)[0]
cv2.rectangle(frame, 
              (startX, startY - 10 - text_size[1] - 10), 
              (startX + text_size[0], startY - 10), 
              color, 
              -1)  # -1 fills the rectangle
cv2.putText(frame, label, (startX, startY - 15),
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)  # White text
```

## 5. Display & User Interface (Lines 155-161)

### Theory
- `cv2.imshow()` displays processed frames
- `cv2.waitKey()` waits for keyboard input (1ms) and returns key code
- `0xFF` masks to get only relevant 8 bits of key code
- `ord("s")` converts character 's' to its ASCII value
- `cv2.destroyAllWindows()` properly closes GUI windows
- `vs.stop()` stops the video stream

### Practical Experience
```python
# Add more keyboard controls
key = cv2.waitKey(1) & 0xFF

if key == ord("s"):
    break
elif key == ord("p"):
    # Pause functionality
    print("Paused - press any key to continue")
    cv2.waitKey(0)  # Wait indefinitely
elif key == ord("c"):
    # Capture screenshot
    timestamp = int(time.time())
    filename = f"screenshot_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved as {filename}")
```

## 6. Face Detection & Mask Classification

### Theory
- The `detect_and_predict_mask()` function processes each frame
- It uses a pre-trained face detection model (from OpenCV's DNN module)
- Detected faces are preprocessed for the mask classifier
- The mask classifier returns confidence scores for "mask" and "no mask"

### Practical Exercise
```python
# Add a confidence threshold slider
cv2.namedWindow("Controls")
confidence_threshold = 0.5
cv2.createTrackbar("Confidence", "Controls", int(confidence_threshold*100), 100, 
                   lambda x: setattr(x, "confidence_threshold", x/100))

# Then in your detection loop:
if confidence > confidence_threshold:
    # Process face...
```

## 7. Debugging Tips

### Practical Troubleshooting
- If no faces are detected, try adjusting lighting or camera position
- To verify model files are loaded correctly, add:
```python
try:
    face_nn = cv2.dnn.readNet(t, w)
    print("Face detection model loaded successfully")
    mask_nn = load_model("mask_detector.h5")
    print("Mask detection model loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()
```

- Check frame shape if facing scaling issues:
```python
print(f"Frame dimensions: {frame.shape}")
```

## 8. Practice Project Extensions

1. **Add face counting statistics:**
   - Track total faces, masked faces, and unmasked faces
   - Display as an overlay on the video

2. **Add recording functionality:**
   - Save video clips when unmasked faces are detected

3. **Implement a simple alarm system:**
   - Play a sound when someone without a mask is detected
   
4. **Add time-based analytics:**
   - Track mask compliance over time
   - Generate simple reports