import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from nlp_pipeline import predict
from cv_pipeline import process_image_with_classes

st.set_page_config(page_title="Video Content Search", layout="wide")

st.title("Video Content Search")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
user_prompt = st.text_input("Enter your prompt")

if uploaded_video and user_prompt:
    with st.spinner("Processing prompt..."):
        class_color_pairs = predict(user_prompt)
        st.success(f"Detected pairs: {class_color_pairs}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.write(uploaded_video.read())
        video_path = tmpfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps)  # 1 frame per second

    frame_results = []
    frame_num = 0

    stframe = st.empty()
    st.subheader("Matching Frames:")

    with st.spinner("Processing video..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                result = process_image_with_classes(frame.copy(), class_color_pairs)
                if not isinstance(result, str):
                    frame_bgr = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    frame_results.append(frame_bgr)

            frame_num += 1

        cap.release()
        os.remove(video_path)

    if frame_results:
        cols = st.columns(6)
        for idx, frame in enumerate(frame_results):
            col = cols[idx % 6]
            with col:
                st.image(frame, caption=f"Frame {idx+1}", use_container_width=True)
    else:
        st.warning("No matching frames found.")