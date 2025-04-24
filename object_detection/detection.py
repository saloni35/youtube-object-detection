import cv2
from utils.utils import *


def detect_objects(model_name, youtube_url, max_frames=100000, skip_frames=1000):
    # Load model
    from ultralytics import YOLO
    model = YOLO(model_name)
    stream_url = get_youtube_stream_url(youtube_url)
    cap = cv2.VideoCapture(stream_url)

    frame_count = 0
    frames_with_detections = []

    while cap.isOpened() and frame_count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)
        annotated_frame = results[0].plot()  # draw boxes
        frames_with_detections.append(annotated_frame)

        frame_count += skip_frames

    cap.release()
    return frames_with_detections

