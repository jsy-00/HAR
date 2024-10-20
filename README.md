# HAR
Human Activity Recognition
![image](https://github.com/user-attachments/assets/032efc3d-bfaf-4bd1-b8dc-45455e241616)
**Human Activity Recognition (HAR) System**
This repository contains the code and resources for a Human Activity Recognition (HAR) system that uses a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architecture. The system is designed to analyze and recognize human activities from video data in real-time.

**Features**
**Hybrid CNN-LSTM Model:** A combination of CNN for spatial feature extraction and LSTM for temporal sequence analysis.
Signal Analysis & Feature Extraction: A Python framework is provided for efficient signal analysis and feature extraction from HAR data streams.
Memory Optimization: Principal Component Analysis (PCA) and batch processing techniques are used to reduce memory consumption during training and inference.
Real-time Deployment: The solution is deployed using Streamlit, allowing users to upload videos and get real-time activity recognition.

**System Overview**
**Model Architecture:**
**CNN:** Extracts spatial features from video frames.
**LSTM:** Captures temporal dependencies and patterns in the extracted features to classify human activities.
**Accuracy:** The system currently achieves a 40% accuracy rate on the test dataset, with potential for further optimization.
**Data Processing:**
Signal preprocessing, feature extraction, and dimensionality reduction using PCA.
Batch processing for handling large datasets efficiently and avoiding memory errors.

**Deployment:**
The model is deployed via a Streamlit web application for easy interaction.
Users can upload their own videos, and the app will perform activity recognition in real time.


**Installation**
To run this project locally, follow these steps:
git clone https://github.com/yourusername/Human-Activity-Recognition.git

Install the required dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

**Usage**
Launch the Streamlit web app.
Upload a video file containing human activity.
The system will process the video and provide real-time activity predictions.


