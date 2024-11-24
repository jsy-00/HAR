import boto3
import cv2
import streamlit as st
import numpy as np
import json

# AWS SageMaker 설정
runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
endpoint_name = "smwu-7-sagemaker-ep-20241124-105946"

# 프레임 처리 함수
def process_frames_with_endpoint(video_path):
    frame_results = []
    frame_index = 1

    # 비디오 읽기
    video_reader = cv2.VideoCapture(video_path)
    total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # 프레임별로 처리
    for i in range(total_frames):
        success, frame = video_reader.read()
        if not success:
            break

        # 프레임을 SageMaker 모델 입력 형식에 맞게 변환
        resized_frame = cv2.resize(frame, (224, 224))  # 모델 입력 크기
        img_encoded = cv2.imencode('.jpg', resized_frame)[1]  # JPG로 인코딩

        # 엔드포인트 호출
        try:
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/x-image",
                Body=img_encoded.tobytes(),  # SageMaker로 전송할 데이터
            )
            result = json.loads(response["Body"].read().decode())
            predicted_class = result["predicted_class"]
        except Exception as e:
            st.error(f"Error processing frame {frame_index}: {e}")
            continue

        # 결과 저장
        frame_results.append(predicted_class)

        # 프레임에 결과 표시
        label = "OK" if predicted_class == 1 else "NG"
        color = (0, 255, 0) if predicted_class == 1 else (0, 0, 255)
        cv2.putText(frame, f"Prediction: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Streamlit으로 프레임 표시
        st.image(frame, channels="BGR", caption=f"Frame {frame_index}: {label}")
        frame_index += 1

    video_reader.release()
    return frame_results

# Streamlit 애플리케이션
st.title("Real-time NG/OK Video Classification")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_file is not None:
    # 업로드된 비디오를 임시 파일로 저장
    temp_video_path = f"temp_{uploaded_file.name}"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(temp_video_path)

    if st.button("Classify Video"):
        with st.spinner("Processing..."):
            results = process_frames_with_endpoint(temp_video_path)
            st.success("Video processing complete!")
            st.write("Frame Results:", results)

        # 임시 파일 삭제
        import os
        os.remove(temp_video_path)
