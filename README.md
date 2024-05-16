# Object Detection System

This project is an object detection system that uses a deep learning model to identify objects in images. The model is trained on a custom dataset and can be easily retrained on new datasets.

## Features

- Train the object detection model on a custom dataset
- Detect objects in uploaded photos with confidence scores
- User-friendly interface for training and detection
- Data augmentation techniques for improved model performance

## Requirements

- Python 3.6 or above
- TensorFlow 2.x
- Keras
- OpenCV
- scikit-learn
- Pillow
- NumPy

## Installation

1. Clone the repository:
   git clone https://github.com/sheydHD/object-detection-CNN.git
   
2. Navigate to the project directory:
  cd object-detection

3. Install the required dependencies:
  pip install -r requirements.txt



## Usage

1. Prepare your dataset:
- Create a directory named "Data" in the project root (if it doesn't exist yet).
- Inside "Data", create a directory named "Train" (if it doesn't exist yet).
- Place your training images in subdirectories within the "Train" directory, where each subdirectory represents a class label.
  (In the given example there are 4 items included: cat, pliers, screwdriver and sword

  
2. Train the model:
- Run the script:
  ```
  python main.py
  ```
- When prompted, choose to train the model by entering "yes".


3. Detect objects in a photo:
- Run the script:
  ```
  python main.py
  ```
- When prompted, choose to upload a photo for object detection by entering "yes".
- Select the photo using the file dialog.
- The script will display the detected object(s) with confidence scores.

