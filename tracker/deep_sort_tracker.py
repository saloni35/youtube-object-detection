from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def track_objects(self, bboxes, confidence, class_names, frame):
        tracks = self.tracker.update_tracks(raw_detections=bboxes, frame=frame)
        tracked_data = []

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            tracked_data.append({
                "bbox": (x1, y1, x2, y2),
                "track_id": track.track_id,
                "class_name": track.get_det_class(),
            })

        return tracked_data
