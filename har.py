import boto3  # AWS S3 SDK
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import ConvLSTM2D, MaxPooling3D, TimeDistributed, Dropout, Flatten, Dense # type: ignore
import numpy as np
import cv2
import streamlit as st
import boto3
from datetime import datetime
import os

# AWS S3 설정
S3_BUCKET = "cv-7-video"
S3_REGION = "us-east-1"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/video/"
s3_client = boto3.client('s3', region_name=S3_REGION)

# NG/OK 분류 클래스
CLASSES_LIST = ['NG', 'OK']

# 사전에 학습된 NG/OK 모델 로드
model = load_model("https://cv-diecasting-7.s3.us-east-1.amazonaws.com/models/")

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

# S3에 비디오 업로드 함수
def upload_video_to_s3(local_file_path):
    # 고유 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    s3_file_name = f"video_ng_ok_{timestamp}.mp4"
    
    # S3로 업로드
    s3_client.upload_file(local_file_path, S3_BUCKET, f"video/{s3_file_name}")
    s3_url = f"{S3_BASE_URL}{s3_file_name}"
    return s3_url, s3_file_name

# S3에서 비디오 다운로드 함수
def download_video_from_s3(s3_file_name):
    local_video_path = f"temp_{s3_file_name}"
    s3_client.download_file(S3_BUCKET, f"video/{s3_file_name}", local_video_path)
    return local_video_path

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
        with st.spinner("Uploading to S3 and Predicting..."):
            # S3에 비디오 업로드
            s3_url, s3_file_name = upload_video_to_s3(temp_file_path)
            st.write(f"Video uploaded to S3: [View Here]({s3_url})")

            # S3에서 비디오 다운로드 후 처리
            local_video_path = download_video_from_s3(s3_file_name)
            predicted_class, all_class_scores = predict_video(local_video_path)

            # 최종 결과 출력
            st.success(f"Predicted Class: {predicted_class}")
            st.write("All Class Scores:")
            for class_name, score in all_class_scores.items():
                st.write(f"{class_name}: {score}")

        # 임시 파일 삭제
        os.remove(temp_file_path)