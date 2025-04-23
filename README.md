# 😷 Real-Time Face Mask Detection

This project implements a real-time face mask detection system using deep learning and computer vision. It detects whether a person is wearing a mask or not using a webcam feed.

## ✨ Features
- 🕒 Real-time face detection and mask classification.
- ⚡ Utilizes MobileNetV2 for efficient and lightweight deep learning.
- 📹 Supports live video streams from a webcam.
- ✅ Provides visual feedback with bounding boxes and labels.

## 📋 Requirements
To run this project, ensure you have the following dependencies installed:

```plaintext
keras==2.11.0
numpy==1.23.5
matplotlib==3.7.1
scikit-learn==1.2.2
imutils==0.5.4
opencv-python==4.8.0.74
tensorflow==2.11.0
```

You can install all dependencies using the following command:
```bash
pip install -r requirements.txt
```

## 📂 Dataset
The project uses a dataset containing images of people with and without masks. The dataset is organized into two categories:
- `with_mask` 😷
- `without_mask` 😮

Ensure the dataset is placed in the `dataset` directory before training the model.

## 🚀 How to Run

### 1. 🏋️ Train the Model
To train the face mask detection model, run the `train_mask_detector.py` script:
```bash
python train_mask_detector.py
```
This will train the model and save it as `mask_detector.model`.

### 2. 🎥 Detect Masks in Real-Time
To start real-time face mask detection, run the `detect_mask_video.py` script:
```bash
python detect_mask_video.py
```
This will open a webcam feed and display bounding boxes around detected faces with labels indicating "Mask" or "No Mask".

### 3. 🛑 Exit the Application
Press the `s` key to stop the webcam feed and exit the application.

## 📁 File Structure
```
Real time face mask detection by me/
├── dataset/                     # Dataset directory
│   ├── with_mask/               # Images of people wearing masks
│   ├── without_mask/            # Images of people without masks
├── train_mask_detector.py       # Script to train the mask detection model
├── detect_mask_video.py         # Script for real-time mask detection
├── requirements.txt             # List of dependencies
├── mask_detector.model          # Trained model (generated after training)
├── README.md                    # Project documentation
```

## 📊 Results
The model achieves high accuracy in detecting masks and provides real-time predictions with minimal latency. The following metrics are displayed during training:
- 📉 Training Loss
- 📉 Validation Loss
- 📈 Training Accuracy
- 📈 Validation Accuracy

## 🖼️ Screenshots
![Mask Detection Example](https://via.placeholder.com/600x400?text=Mask+Detection+Example)

## 🙏 Acknowledgments
This project is inspired by the need for mask detection during the COVID-19 pandemic. It leverages the power of deep learning and computer vision to provide a practical solution.

## 📜 License
This project is licensed under the MIT License. Feel free to use and modify it as needed.

## 📬 Contact
For any questions or feedback, feel free to reach out:
- **📧 Email**: brajeshguptaa1@example.com
- **🐙 GitHub**: [LazyCoderForU](https://github.com/your-profile)

Happy coding! 🚀