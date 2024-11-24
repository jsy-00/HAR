import tensorflow as tf
import requests
import os
import tarfile
import boto3
import numpy as np
import cv2
import streamlit as st

# AWS S3 설정
S3_BUCKET = "cv-diecasting-7"
S3_REGION = "us-east-1"
MODEL_KEY = "models/efficientnet/model.tar.gz"
LOCAL_MODEL_DIR = "model_directory"

# NG/OK 분류 클래스
CLASSES_LIST = ['NG', 'OK']

# 모델 다운로드 함수 (S3에서 다운로드)
def download_model_from_s3():
    if not os.path.exists(LOCAL_MODEL_DIR):  # 모델 디렉토리가 없는 경우 생성
        os.makedirs(LOCAL_MODEL_DIR)
    if not os.path.exists(os.path.join(LOCAL_MODEL_DIR, "saved_model")):  # 모델 디렉토리에 저장된 모델이 없는 경우
        with st.spinner("Downloading model from S3..."):
            s3_client = boto3.client('s3', region_name=S3_REGION)
            tar_path = os.path.join(LOCAL_MODEL_DIR, "model.tar.gz")
            with open(tar_path, 'wb') as f:
                s3_client.download_fileobj(S3_BUCKET, MODEL_KEY, f)
            # 압축 해제
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=LOCAL_MODEL_DIR)
            st.success("Model downloaded and extracted successfully!")
    else:
        st.info("Model already exists locally.")

# 모델 로드 함수
def load_trained_model():
    try:
        download_model_from_s3()
        model = tf.keras.models.load_model(os.path.join(LOCAL_MODEL_DIR, "saved_model"))  # TensorFlow의 SavedModel 로드
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error("Failed to load the model. Please check the model file or URL.")
        print(f"Error loading model: {e}")
        return None

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
    if model is None:
        st.error("Model is not loaded. Prediction cannot be performed.")
        return None, None
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

            if predicted_class and all_class_scores:
                st.success(f"Predicted Class: {predicted_class}")
                st.write("All Class Scores:")
                for class_name, score in all_class_scores.items():
                    st.write(f"{class_name}: {score}")

        # 임시 파일 삭제
        os.remove(temp_file_path)
