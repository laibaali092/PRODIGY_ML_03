# PRODIGY_ML_03

# Objective: 
The goal of this project is to develop a machine learning model that can accurately identify and classify different hand gestures using image data. This model enables intuitive human-computer interaction and can be used for gesture-based control systems.

# Dataset:

Source: Kaggle - LeapGestRecog Dataset
Structure: The dataset is organized into folders corresponding to different gestures. Each gesture folder contains subfolders with images representing various instances of the gesture.
Tools & Libraries:

Python Libraries: os, cv2, numpy, matplotlib
Machine Learning: sklearn for data preprocessing and splitting
Deep Learning: TensorFlow, Keras for building the Convolutional Neural Network (CNN) model
Project Workflow:

# Mount Google Drive:

Google Drive is mounted in Google Colab to access the dataset directly from your drive.
Load and Preprocess Data:

Image Loading: Images are loaded from the dataset folders, converted to grayscale, and resized to a standard size (64x64 pixels).
Normalization: The images are normalized by dividing pixel values by 255.0.
Label Encoding: Labels are encoded into one-hot vectors to be used in classification.
Split the Dataset:

The dataset is split into training and testing sets using an 80-20 split ratio.
Build the CNN Model:

# Model Architecture:
Conv2D Layers: Extract features from the images.
MaxPooling2D Layers: Reduce the spatial dimensions of the feature maps.
Flatten Layer: Convert the 2D matrices to 1D vectors.
Dense Layers: Perform classification based on the features.
Dropout Layer: Prevent overfitting by randomly setting some activations to zero during training.
Compilation: The model is compiled using the Adam optimizer and categorical crossentropy loss.

# Train the Model:

The model is trained using the training data with early stopping to prevent overfitting.
Early Stopping: Monitors validation loss and stops training when the performance ceases to improve.
Evaluate the Model:

The model is evaluated on the test data to determine its accuracy.
Accuracy and Loss: Training and validation accuracy/loss are plotted to visualize the model's performance.

# Save the Model:

The trained model is saved to Google Drive for future use.
Challenges and Solutions:

Directory Path Errors: Ensure the correct path to the dataset and verify the directory structure.
Handling Missing or Empty Directories: The code includes checks to handle scenarios where directories are missing or empty, ensuring the model doesn't crash during data loading.
Outcome: The CNN model is successfully trained and evaluated, achieving an accuracy score on the test set. The trained model is saved, enabling it to be used in real-world applications like gesture-based controls and HCI systems.

This project demonstrates how to develop a robust gesture recognition system using deep learning techniques, handling real-world data issues, and effectively deploying the model for practical use.
