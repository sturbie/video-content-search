import streamlit as st
import tempfile

import os
os.environ["STREAMLIT_WATCH_FOR_CHANGES"] = "false"

import cv2
import numpy as np

#Download from Google Drive
import gdown
import zipfile
import subprocess

def download_file_with_curl(file_id, destination):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    print(f"Trying curl fallback for: {destination}")
    try:
        subprocess.run(["curl", "-L", url, "-o", destination], check=True)
    except subprocess.CalledProcessError as e:
        print(f"curl failed: {e}")

def download_file_from_google_drive(file_id, destination):
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        print(f"Trying gdown for: {destination}")
        gdown.download(url, destination, quiet=False)
    except Exception as e:
        print(f"gdown failed to download {destination}. Error: {e}")
        download_file_with_curl(file_id, destination)

def unzip_if_needed(file_path):
    if file_path.endswith(".zip") and os.path.isfile(file_path):
        try:
            print(f"Unzipping {file_path}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                extract_path = os.path.splitext(file_path)[0]
                zip_ref.extractall(extract_path)
            print(f"Extracted to: {extract_path}")
        except zipfile.BadZipFile as e:
            print(f"Failed to unzip {file_path}. Error: {e}")

FILES_TO_DOWNLOAD = {
    "nlp_cv_ner_model.zip": "17VC58QyfUW89A10SfeqOJNiEzqYRPnA9",
    "yolov4-tiny.weights": "1jnMvCvtvTYBBf62-uH1OpG1wzQmCTk4W"
}

for filename, file_id in FILES_TO_DOWNLOAD.items():
    if not os.path.exists(filename):
        print(f"\nDownloading {filename}...")
        download_file_from_google_drive(file_id, filename)
        if filename.endswith(".zip"):
            unzip_if_needed(filename)

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
