import cv2
from utils.utils import *


def compare_models_in_detection(model_names, youtube_url, max_frames=100000, skip_frames=1000):
    # Load model
    from ultralytics import YOLO
    models = {name: YOLO(name) for name in model_names}
    stream_url = get_youtube_stream_url(youtube_url)
    cap = cv2.VideoCapture(stream_url)

    frame_count = 0
    frames_with_detections = []
    model_outputs = {name: [] for name in model_names}

    while cap.isOpened() and frame_count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        for name, model in models.items():
            result = model(frame)
            model_outputs[name].append(result[0].plot())

        frame_count += skip_frames

    cap.release()
    return model_outputs

