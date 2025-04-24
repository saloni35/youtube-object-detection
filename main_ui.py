
import streamlit as st
from PIL import Image
from object_detection.tracking import *
from object_detection.detection import *
from object_detection.compare_yolo_models_detection import *


st.set_page_config(page_title="YouTube YOLO Detection", layout="centered")
st.title("🎥 YouTube Object Detection using YOLO models")
st.markdown("Paste a YouTube video URL, and we’ll detect objects on a few frames.")

# Get YouTube URL from user
youtube_url = st.text_input("Enter YouTube URL:")

task = st.selectbox(
    "Choose Task:",
    [
        "Just Detection",
        "Detection and Tracking",
        "Compare YOLO Models"
    ]
)

model_options = ["yolov5s.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
selected_model = None
selected_models = None
if task == "Compare YOLO Models":
    selected_models = st.multiselect("Select Models to Compare", model_options, default=["yolov8n.pt"])
else:
    selected_model = st.selectbox("Select Models to Compare", model_options)

perform_task = st.button("Perform Task")

if "generated_imgs" not in st.session_state:
    st.session_state["generated_imgs"] = []

if "generated_multi_imgs" not in st.session_state:
    st.session_state["generated_multi_imgs"] = []

# Run detection
if perform_task and task != "Compare YOLO Models":
    if youtube_url and len(selected_model) != 0:
        st.session_state["generated_imgs"] = []
        st.session_state["generated_multi_imgs"] = []
        if task == "Just Detection":
            with st.spinner("Fetching video and running YOLO detection..."):
                frames = detect_objects(selected_model, youtube_url, max_frames=1000, skip_frames=100)
            st.success("Detection complete!")
            for idx, frame in enumerate(frames):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(Image.fromarray(rgb), caption=f"Detected Frame {idx + 1}")
                st.session_state["generated_imgs"].append((Image.fromarray(rgb), f"Detected Frame {idx + 1}"))

        elif task == "Detection and Tracking":
            with st.spinner("Fetching video and running YOLO detection and tracking..."):
                frames = track_objects(selected_model, youtube_url, max_frames=1000, skip_frames=100)
            st.success("Detection complete!")
            for idx, frame in enumerate(frames):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(Image.fromarray(rgb), caption=f"Tracked Frame {idx + 1}")
                st.session_state["generated_imgs"].append((Image.fromarray(rgb), f"Tracked Frame {idx + 1}"))
    else:
        st.warning("Please enter a valid YouTube URL and select a model")

if task == "Compare YOLO Models":
    if perform_task:
        if youtube_url and len(selected_models) != 0:
            st.session_state["generated_imgs"] = []
            st.session_state["generated_multi_imgs"] = []
            with st.spinner("Fetching video and running YOLO detection with chosen YOLO models..."):
                frames = compare_models_in_detection(selected_models, youtube_url, max_frames=1000, skip_frames=100)
            st.success("Detection complete!")
            for idx, frame in enumerate(frames[selected_models[0]]):
                st.subheader(f"Frame {idx + 1}")
                row_imgs = []
                cols = st.columns(len(selected_models))
                for col, model in zip(cols, selected_models):
                    with col:
                        rgb = cv2.cvtColor(frames[model][idx], cv2.COLOR_BGR2RGB)
                        st.image(Image.fromarray(rgb), caption=model)
                        row_imgs.append((Image.fromarray(rgb), model))
                st.session_state["generated_multi_imgs"].append(row_imgs)
        else:
            st.warning("Please enter a valid YouTube URL and select atleast one model")

if st.session_state["generated_imgs"]:
    for img, caption in st.session_state["generated_imgs"]:
        st.image(img, caption=caption)

if st.session_state["generated_multi_imgs"]:
    for idx, row_imgs in enumerate(st.session_state["generated_multi_imgs"]):
        st.subheader(f"Frame {idx + 1}")
        cols = st.columns(len(row_imgs))
        for col, (img, model) in zip(cols, row_imgs):
            with col:
                st.image(img, caption=model)
