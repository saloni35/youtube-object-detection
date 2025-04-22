import cv2
import yt_dlp



def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'quiet': True,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict['url']


def detect_objects_from_youtube(youtube_url, max_frames=100000, skip_frames=1000):
    # Load model
    from ultralytics import YOLO
    model = YOLO("yolov8s.pt")  # You can use yolov5s.pt, yolov8m.pt, etc.
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

