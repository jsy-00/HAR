import boto3
import cv2
import streamlit as st
import numpy as np
import json
from dotenv import load_dotenv
import os
# 더미 데이터 용
import random

# .env 파일 로드
load_dotenv()

# 이미지 해시 함수
def get_image_hash(image):
    resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    avg = gray.mean()
    return ''.join('1' if pixel > avg else '0' for row in gray for pixel in row)

# 해밍 거리 계산 함수
def hamming_distance(hash1, hash2):
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

# 영상 처리 함수
def process_video(video_path, tolerance=5):
    cap = cv2.VideoCapture(video_path)
    prev_hash = None
    unique_images = []
    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 영상 끝에 도달한 경우
        
        # 현재 프레임의 해시 값 계산
        current_hash = get_image_hash(frame)
        
        if prev_hash is None or (tolerance < hamming_distance(prev_hash, current_hash) < 40):
            unique_images.append(frame)  # 고유 이미지 저장
        
        prev_hash = current_hash
        frame_index += 1
        progress_bar.progress(frame_index / total_frames)
    
    cap.release()
    progress_bar.empty()
    return unique_images

# SageMaker 호출 함수
def invoke_sagemaker_endpoint(endpoint_name, image):
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION")
    runtime = boto3.client("sagemaker-runtime",
                           aws_access_key_id=aws_access_key,
                           aws_secret_access_key=aws_secret_key,
                           region_name=aws_region)
    _, img_encoded = cv2.imencode(".jpg", image)
    payload = img_encoded.tobytes()
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/x-image",
        Body=payload
    )
    result = json.loads(response["Body"].read().decode())
    return result["predicted_class"]

# 이미지 테두리 추가
def add_border(image, color, border_thickness=50):
    return cv2.copyMakeBorder(
        image,
        top=border_thickness,
        bottom=border_thickness,
        left=border_thickness,
        right=border_thickness,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )
    
# 결과 표시 함수 (5개씩 묶어서 표시)
def display_results(unique_images, results):
    st.subheader("Predict Result")
    
    ng_images = []
    ng_No = set()
    ok_No = set()

    # 5개씩 묶어서 처리
    for i in range(0, len(results), 5):
        batch_results = results[i:i+5]  # 현재 묶음 결과
        batch_images = unique_images[i:i+5]  # 현재 묶음 이미지
        
        # 부품 상태 결정 (하나라도 NG이면 전체 NG)
        batch_status = "NG" if 0 in batch_results else "OK"
        color = (0, 255, 0) if batch_status == "OK" else (0, 0, 255)
        
        # 최종 결과 표시용
        part_number = i // 5 + 1
        if batch_status == "NG":
            ng_No.add(part_number)
        else:
            ok_No.add(part_number)

        # 부품 상태 표시
        st.markdown(f"### No. {i//5 + 1}: **{batch_status}**")

        cols = st.columns(5)
        # 각 이미지에 결과 표시
        for j, (image, status) in enumerate(zip(batch_images, batch_results)):
            label = "OK" if status == 1 else "NG"
            label_color = (0, 255, 0) if status == 1 else (0, 0, 255)
            
            # NG 이미지 저장
            if status == 0:
                ng_images.append((image, i + j))
            
            bordered_image = add_border(image, label_color)
            part_number = (i + j) // 5 + 1
            channel_number = (i + j) % 5 + 1
            cols[j].image(
                bordered_image,
                channels="BGR",
                caption=f"Part {part_number} - Channel {channel_number} ({label})",
            )
    # NG 이미지만 추가 출력
    if ng_images:
        st.subheader("Final NG Images")
        cols = st.columns(5)
        for idx, (ng_image, ng_index) in enumerate(ng_images):
            part_number = ng_index // 5 + 1
            channel_number = ng_index % 5 + 1
            
            bordered_ng_image = add_border(ng_image, (0, 0, 255))
            
            # 이미지 출력: 부품 번호와 채널 번호 표시
            cols[idx % 5].image(
                bordered_ng_image, 
                channels="BGR", 
                caption=f"No. {part_number} - Channel {channel_number}"
            )
    
    # 최종 NG/OK 부품 번호 출력
    st.subheader("Final Result Summary")
    if ng_No:
        st.error(f"NG Parts: {','.join(map(str, ng_No))} (Total: {len(ng_No)})")
    if ok_No:
        st.success(f"OK Parts: {','.join(map(str, ok_No))} (Total: {len(ok_No)})")
            

# 메인 함수
def main():
    # Streamlit 애플리케이션
    st.title("Real-time NG/OK Video Classification")
    
    # 고유 프레임 저장용 세션 상태 초기화
    if "unique_images" not in st.session_state:
        st.session_state["unique_images"] = []
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # 업로드된 비디오를 임시 파일로 저장
        temp_video_path = f"temp_{uploaded_file.name}"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Complete Upload File : {uploaded_file.name}")
    
        # 업로드 출력
        st.subheader("Uploaded Video")
        st.video(temp_video_path)
    
        # 영상 이미지 추출
        if st.button("Abstract Image from Video"):
            with st.spinner("Processing..."):
                unique_images = process_video(temp_video_path, tolerance=5)
                st.session_state["unique_images"] = unique_images  # 세션 상태에 저장
                st.success(f"Complete: {len(unique_images)} Images Abstracted")
            
            # # 고유 프레임 확인
            # st.subheader("Abstracted Images")
            # for i, frame in enumerate(st.session_state["unique_images"]):
            #     st.image(frame, channels="BGR", caption=f"Frame {i+1}")
        
        # SageMaker 분석
        st.subheader("Start SageMaker Inference")
        endpoint_name = "test-endpoint"
        
        # SageMaker 분석
        if st.button("Start Inference"):
            if not st.session_state["unique_images"]:  # 고유 프레임이 없는 경우 에러 처리
                st.error("No frames available. Please extract frames first!")
            else:
                results = []
                for i, image in enumerate(st.session_state["unique_images"]):
                    with st.spinner(f"Image {i+1}/{len(st.session_state['unique_images'])} processing..."):
                        result = invoke_sagemaker_endpoint(endpoint_name, image)
                        results.append(result)
                st.success("Inference Complete!")
                display_results(st.session_state["unique_images"], results)
                

# 프로그램 실행
if __name__ == "__main__":
    main()
