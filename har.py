import requests
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ConvLSTM2D, MaxPooling3D, TimeDistributed, Dropout, Flatten, Dense
import numpy as np
import cv2
import streamlit as st
from datetime import datetime

# HTTPS 모델 URL
MODEL_URL = "https://cv-diecasting-7.s3.us-east-1.amazonaws.com/models/video_ng_ok.h5"
LOCAL_MODEL_PATH = "video_ng_ok.h5"  # 로컬에 저장할 모델 파일명

# NG/OK 분류 클래스
CLASSES_LIST = ['NG', 'OK']

# HTTPS URL에서 모델 다운로드 함수
def download_model_from_url():
    if not os.path.exists(LOCAL_MODEL_PATH):  # 이미 로컬에 모델 파일이 없는 경우
        with st.spinner("Downloading model from URL..."):
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(LOCAL_MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:  # 빈 내용 방지
                            f.write(chunk)
                st.success("Model downloaded successfully!")
            else:
                st.error(f"Failed to download model. Status code: {response.status_code}")
                raise Exception("Model download failed")
    else:
        st.info(f"Model already exists locally at {LOCAL_MODEL_PATH}")

# 모델 로드 함수
def load_trained_model():
    download_model_from_url()
    model = load_model(LOCAL_MODEL_PATH)  # 로컬에 저장된 모델 로드
    return model

# 모델 로드
model = load_trained_model()

# 비디오 전처리 함수
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

# 비디오 처리 및 분류 함수
def predict_video(video_path):
    preprocessed_frames = preprocess_frames(video_path)
    preprocessed_frames = np.expand_dims(preprocessed_frames, axis=0)
    predictions = model.predict(preprocessed_frames)
    all_class_scores = {CLASSES_LIST[i]: predictions[0][i] for i in range(len(CLASSES_LIST))}
    predicted_class_index = np.argmax(predictions)
    predicted_class = CLASSES_LIST[predicted_class_index]
    return predicted_class, all_class_scores

# Streamlit 애플리케이션
st.title("Diecasting NG/OK Classification")
st.write("Upload a diecasting video to classify as NG (Defective) or OK (Good Quality)")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_file is not None:
    # 로컬에 업로드된 비디오 저장
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video(temp_file_path)  # 업로드된 비디오 표시

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            # 로컬 파일로 처리
            predicted_class, all_class_scores = predict_video(temp_file_path)

            # 최종 결과 출력
            st.success(f"Predicted Class: {predicted_class}")
            st.write("All Class Scores:")
            for class_name, score in all_class_scores.items():
                st.write(f"{class_name}: {score}")

        # 임시 파일 삭제
        os.remove(temp_file_path)
