# Object Detection System

This project is an object detection system that uses a deep learning model to identify objects in images. The model is trained on a custom dataset and can be easily retrained on new datasets.

## Features

- Train the object detection model on a custom dataset
- ![image](https://github.com/user-attachments/assets/d16a0aba-91a9-4162-9825-cf45eb434377)
- Detect objects in uploaded photos with confidence scores
- ![image](https://github.com/user-attachments/assets/fa2c788f-d9ea-4499-828f-cf957cfc636b)
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
   ```
   git clone https://github.com/sheydHD/object-detection-CNN.git
   ```
   
3. Navigate to the project directory:
   ```
   cd object-detection
   ```
   
5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```


## Usage

1. Prepare your dataset:
- Create a directory named "Data" in the project root (if it doesn't exist yet).
- Inside "Data", create a directory named "Train" (if it doesn't exist yet).
- Place your training images in subdirectories within the "Train" directory, where each subdirectory represents a class label.
  (In the given example there are 4 items included: cats, pliers, screwdrivers and swords)

  
2. Train the model:
- Run the script:
  ```
  python main.py
  ```
- When prompted, train the model by entering "yes" or skip that part with "no".
  ![image](https://github.com/user-attachments/assets/3ceae83f-c85e-4999-92aa-90808152d96b)

- The code will train the model with the training objects you placed in Data/Train.
- After it finishes, you will receive a trained model file: _best_object_detector_model.keras_ and the model will ask you to select a photo for detection.
  ![image](https://github.com/user-attachments/assets/c3bf7d4b-8807-431f-81be-5d29d80e3d74)

3. Detect objects in a photo:
- Run the script:
  ```
  python main.py
  ```
  
- When prompted, choose to upload a photo for object detection by entering "yes".
- Select the photo using the file dialogue.
- The script will display the detected object(s) with confidence scores.
  ![image](https://github.com/user-attachments/assets/6e070e56-c6e5-4104-a7a7-ba2c94ee096a)


### Bonus
- In the repository, I also included a pre-trained model for detecting the 4 objects mentioned above: _best_object_detector_model_advanced.keras_.
- If you wish to test it, rename the file to _best_object_detector_model.keras_ (remember to rename your previous model if you have one).
  
