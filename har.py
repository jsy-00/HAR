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

# 커스텀 객체로 모델 로드에 필요한 레이어 및 구성 요소 정의
custom_objects = {'ConvLSTM2D': ConvLSTM2D, 'MaxPooling3D': MaxPooling3D, 'TimeDistributed': TimeDistributed,
                  'Dropout': Dropout, 'Flatten': Flatten, 'Dense': Dense}

# 사전에 학습된 모델 로드
model = load_model("video.h5", custom_objects=custom_objects)

# 예측 가능한 클래스 리스트 정의
CLASSES_LIST = ['Archery', 'BabyCrawling', 'BenchPress', 'Biking', 'Bowling', 'FrontCrawl',
                'HandstandPushups', 'HeadMassage', 'HighJump', 'HorseRace', 'JavelinThrow',
                'LongJump', 'PlayingPiano', 'PullUps', 'Punch', 'Shotput', 'SkyDiving', 'SoccerPenalty',
                'SumoWrestling', 'Surfing', 'Typing'
                ]

# 비디오 프레임 전처리 함수 정의
def preprocess_frames(video_path, sequence_length=25):
    frames_list = []  # 프레임을 저장할 리스트
    video_reader = cv2.VideoCapture(video_path)  # 비디오 파일 로드
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))  # 비디오의 총 프레임 수 계산
    skip_frames_window = max(int(video_frames_count / sequence_length), 1)  # 시퀀스 길이에 맞게 프레임 간격 설정
    
    for frame_counter in range(sequence_length):
        # 지정된 위치의 프레임 읽기
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:  # 읽기 실패 시 중단
            break
        resized_frame = cv2.resize(frame, (64, 64))  # 프레임 크기 조정
        normalized_frame = resized_frame / 255.0  # 프레임 정규화
        frames_list.append(normalized_frame)
    
    video_reader.release()  # 비디오 리더 닫기
    return np.array(frames_list)  # 프레임 배열 반환

# 비디오 예측 함수 정의
def predict_video(video_path):
    preprocessed_frames = preprocess_frames(video_path)  # 비디오 전처리
    preprocessed_frames = np.expand_dims(preprocessed_frames, axis=0)  # 모델 입력 형태에 맞게 변환
    predictions = model.predict(preprocessed_frames)  # 모델로 예측 실행
    predicted_class_index = np.argmax(predictions)  # 예측 결과 중 가장 높은 확률의 클래스 인덱스 선택
    predicted_class = CLASSES_LIST[predicted_class_index]  # 예측된 클래스명 가져오기
    return predicted_class, predictions

# Streamlit 애플리케이션 UI 구성
st.title("SMU Diecasting Classification")  # 제목 설정
st.write("Upload a video file to predict NG or OK")  # 설명 텍스트 출력

uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])  # 파일 업로드 컴포넌트

if uploaded_file is not None:
    st.video(uploaded_file)  # 업로드된 비디오 표시

    if st.button("Predict"):  # "Predict" 버튼 클릭 시 실행
        with st.spinner('Predicting...'):  # 로딩 메시지 표시
            # 업로드된 비디오를 임시 파일로 저장
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 비디오 클래스 예측 실행
            predicted_class, predictions = predict_video("temp_video.mp4")

            # 상위 5개의 예측 클래스와 점수 계산
            top_5_indices = np.argsort(predictions[0])[::-1][:5]
            top_5_classes = [CLASSES_LIST[i] for i in top_5_indices]
            top_5_scores = [predictions[0][i] for i in top_5_indices]

            # 예측 결과 출력
            st.success(f"Predicted Class: {predicted_class}")  # 최종 예측 클래스 출력
            st.write("Top 5 Probable Classes")  # 상위 5개 클래스 출력
            for class_name, score in zip(top_5_classes, top_5_scores):
                st.write(f"{class_name}: {score}")  # 클래스와 점수 출력
