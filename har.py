from keras.layers import ConvLSTM2D, MaxPooling3D, TimeDistributed, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import random
from keras.models import load_model
import streamlit as st

import matplotlib
matplotlib.use('Agg')

# Load the trained model with custom objects
custom_objects = {'ConvLSTM2D': ConvLSTM2D, 'MaxPooling3D': MaxPooling3D, 'TimeDistributed': TimeDistributed,
                  'Dropout': Dropout, 'Flatten': Flatten, 'Dense': Dense}

# Load the trained model
model = load_model("video.h5",custom_objects=custom_objects)

# List of classes
CLASSES_LIST = ['Archery', 'BabyCrawling','BenchPress','Biking','Bowling','FrontCrawl',
                'HandstandPushups','HeadMassage','HighJump','HorseRace','JavelinThrow',
                'LongJump', 'PlayingPiano','PullUps','Punch','Shotput','SkyDiving','SoccerPenalty',
                'SumoWrestling','Surfing','Typing'
                ]

def preprocess_frames(video_path, sequence_length=25):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / sequence_length), 1)
    
    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (64, 64))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)
    
    video_reader.release()
    return np.array(frames_list)

def predict_video(video_path):
    preprocessed_frames = preprocess_frames(video_path)
    # Reshape frames to match the model input shape
    preprocessed_frames = np.expand_dims(preprocessed_frames, axis=0)
    # Predict using the model
    predictions = model.predict(preprocessed_frames)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)
    # Get the predicted class name
    predicted_class = CLASSES_LIST[predicted_class_index]
    return predicted_class, predictions

# Streamlit app
st.title("SMU Diecasting Classification")
st.write("Upload a video file to predict the human activity.")

uploaded_file = st.file_uploader("Choose a video file", type=["Avi"])

if uploaded_file is not None:
    # Display the video
    st.video(uploaded_file)

    # Predict when the user clicks the button
    if st.button("Predict"):
        with st.spinner('Predicting...'):
            # Save the uploaded video locally
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Predict the class of the video
            predicted_class, predictions = predict_video("temp_video.mp4")

            top_5_indices = np.argsort(predictions[0])[::-1][:5]
            top_5_classes = [CLASSES_LIST[i] for i in top_5_indices]
            top_5_scores = [predictions[0][i] for i in top_5_indices]

            # Display the top 5 predicted classes and their confidence scores
            st.success(f"Predicted Class: {predicted_class}")
            st.write("Top 5 Probable Classes")
            for class_name, score in zip(top_5_classes, top_5_scores):
                st.write(f"{class_name}: {score}")




