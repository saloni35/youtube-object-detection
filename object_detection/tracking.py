import cv2
from tracker.deep_sort_tracker import Tracker
from utils.utils import *


def track_objects(model_name, youtube_url, max_frames=100000, skip_frames=1000):
    # Load model
    from ultralytics import YOLO
    model = YOLO(model_name)
    stream_url = get_youtube_stream_url(youtube_url)
    cap = cv2.VideoCapture(stream_url)

    tracker = Tracker()
    frame_count = 0
    tracked_frames = []

    while cap.isOpened() and frame_count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        result = model(frame)
        results = result[0]

        bboxes, confs, names = [], [], []
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            bboxes.append([x1, y1, x2 - x1, y2 - y1])
            confs.append(conf)
            names.append(model.names[cls])

        for bbox, conf, cls_name in zip(bboxes, confs, names):
            detections.append([bbox, conf, cls_name])

        if detections and frame.any():
            tracked = tracker.track_objects(detections, confs, names, frame)

            for obj in tracked:
                x1, y1, x2, y2 = obj["bbox"]
                track_id = obj["track_id"]
                label = obj["class_name"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            tracked_frames.append(frame)

        frame_count += skip_frames

    cap.release()
    return tracked_frames

