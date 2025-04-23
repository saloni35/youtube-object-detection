
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from object_detection.tracking import detect_objects_from_youtube


st.set_page_config(page_title="YouTube YOLO Detection", layout="centered")
st.title("🎥 YouTube Object Detection using YOLOv8")
st.markdown("Paste a YouTube video URL, and we’ll detect objects on a few frames.")

# Get YouTube URL from user
youtube_url = st.text_input("Enter YouTube URL:")

# Run detection

if st.button("Run Detection"):
    if youtube_url:
        with st.spinner("Fetching video and running YOLO detection..."):
            frames = detect_objects_from_youtube(youtube_url, max_frames=10000, skip_frames=100)

        st.success("Detection complete!")
        for idx, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(Image.fromarray(rgb), caption=f"Detected Frame {idx + 1}")
    else:
        st.warning("Please enter a valid YouTube URL.")
