# Real-Time Face Mask Detector

## 📝 Description
The Real-Time Face Mask Detector is a computer vision project that detects faces in real-time and determines whether the person is wearing a mask or not. It uses deep learning models for face detection and mask classification.

### Key Features
- Real-time face detection using OpenCV's DNN module.
- Mask detection using a pre-trained deep learning model.
- Displays bounding boxes and labels ("Mask" or "No Mask") on detected faces.

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/real-time-face-mask-detector.git
   cd real-time-face-mask-detector
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Usage
To run the real-time face mask detection:
```bash
python detect.py
```

### Expected Output
- A video stream window will open, showing the real-time detection of faces with labels indicating "Mask" or "No Mask".
- Press the `s` key to stop the video stream.

## 🖼️ Screenshots / Demo
![Demo Screenshot](path/to/screenshot.png)

## 🧠 Tech Stack / Built With
- Python
- OpenCV
- Keras
- NumPy
- imutils

## 📂 Project Structure
```
real-time-face-mask-detector/
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── detect.py
├── train.py
├── train_mask_detector.py
├── mask_detector.h5
├── requirements.txt
├── README.md
└── ...
```

## 🙌 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Open a pull request.

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 👨‍💻 Author
- Your Name
- [GitHub Profile](https://github.com/yourusername)
- [LinkedIn Profile](https://linkedin.com/in/yourprofile)

## 🌐 Links
- [Project Repository](https://github.com/yourusername/real-time-face-mask-detector)