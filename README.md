Eye Disease Detection Using Ensemble Deep Learning

This project is an AI-powered web application developed for automatic eye disease detection using ensemble deep learning models. The system analyzes retinal fundus images and predicts different eye diseases with high accuracy by combining multiple pretrained CNN models.

The application is built using Python, Flask, TensorFlow, Keras, and OpenCV. It uses an ensemble learning approach where predictions from several deep learning architectures are combined to improve classification performance and reduce prediction errors.

Features
Eye disease prediction from retinal images
Ensemble deep learning model implementation
Real-time prediction through web interface
Prediction confidence score display
Prediction history storage using SQLite database
User-friendly dashboard interface
Multiple disease classification support
Diseases Detected

The system can classify the following eye diseases:

Age-related Macular Degeneration (AMD)
Cataract
Diabetic Retinopathy
Glaucoma
Normal Eye Condition
Deep Learning Models Used

The project combines outputs from multiple pretrained CNN models:

DenseNet121
ResNet50
MobileNetV2
VGG16
Inception Network

The final prediction is generated using ensemble averaging, which improves accuracy and model stability.

Technologies Used
Python
Flask
TensorFlow
Keras
OpenCV
NumPy
SQLite
HTML/CSS/JavaScript
Working Process
User uploads a retinal eye image.
The image is resized and preprocessed.
Feature extraction is performed using pretrained CNN models.
Multiple classifiers generate predictions.
Ensemble averaging combines all outputs.
Final disease prediction and confidence score are displayed.
Results are stored in the database for history tracking.
