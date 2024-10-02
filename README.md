Hand Gesture Recognition with HandTrackingModule

This project captures hand gestures using a webcam, processes the images, and classifies the gestures using a trained machine learning model. It uses OpenCV and cvzone.HandTrackingModule for real-time hand tracking, and TensorFlow for gesture classification.

Features
Real-time hand tracking: Detect hands using a webcam.
Hand cropping and resizing: Process and normalize hand images for model input.
Gesture classification: Use a trained TensorFlow model to classify hand gestures.
Save cropped hand images: Captures and saves hand images for dataset collection.
Requirements
Make sure to install the following dependencies before running the project:

bash

  pip install opencv-python
  pip install cvzone
  pip install numpy
  pip install tensorflow
  Installation
  Clone the repository to your local machine:
  bash
  Copy code
  git clone https://github.com/username/HandGestureRecognition.git
  Navigate to the project directory:
  bash
  Copy code
  cd HandGestureRecognition
  Install the necessary libraries:
  bash
  Copy code
  pip install -r requirements.txt

If requirements.txt is not available, manually install the dependencies listed above.

Usage
Data Collection (Optional): If you want to collect new hand gesture data, modify the folder variable to specify the save location. Press s to save a hand image during the session.

Running the Real-Time Hand Tracking: Execute the Python script to start tracking hands and classifying gestures.

bash

  python HandTrackingModule.py
  Model Integration: To classify gestures using a TensorFlow model:

Train your model using Teachable Machine or TensorFlow.
Download and place the model in your project directory.
Modify the script to load your TensorFlow model.

Training a Model:

Collect hand images using the provided script.
Train the model using Teachable Machine or TensorFlow.
Integrate the trained model into your system for real-time gesture recognition.

How It Works

1. Hand Tracking:
The script uses cvzone.HandTrackingModule to detect and crop hand images from the webcam feed.

2. Gesture Classification:
A TensorFlow model is loaded to classify gestures based on the preprocessed hand image. The model takes the normalized hand image, predicts the gesture, and displays the result in real-time on the webcam feed.

Directory Structure
├── HandTrackingModule.py     # Main script for hand tracking and gesture recognition
├── Data/                     # Folder to store collected hand images
├── model/                    # Pre-trained TensorFlow model (after training)
├── README.md                 # Project documentation

Future Improvements
Expand the dataset to improve the accuracy of gesture recognition.
Add more complex gesture recognition logic.
Explore real-time applications, such as controlling devices using hand gestures.
