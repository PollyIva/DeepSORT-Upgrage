"""Video processing, visualization and MOT output."""

import os
import time
import cv2


class MOTWriter:
    def __init__(self, filename):
        self.file = open(filename, "w")


    def write(self, frame_id, track_id, bbox):
        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        line = (
            f"{frame_id},{track_id},"
            f"{x1:.2f},{y1:.2f},"
            f"{width:.2f},{height:.2f},"
            "1,-1,-1,-1\n"
        )

        self.file.write(line)


    def close(self):
        self.file.close()


def draw_tracks(frame, tracks):

    for track in tracks:

        x1, y1, x2, y2 = map(int, track["bbox"])

        track_id = track["id"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    return frame


def run_sequence(
    img_dir,
    output_video,
    mot_path,
    detector,
    reid,
    tracker,
    fps=30
):

    frames = sorted(os.listdir(img_dir))

    first_frame = cv2.imread(os.path.join(img_dir, frames[0]))
    h, w = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    mot_writer = MOTWriter(mot_path)

    total_time = 0

    for i, frame_name in enumerate(frames, start=1):

        frame_path = os.path.join(img_dir, frame_name)
        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        start = time.time()

        detections = detector.detect(frame)
        tracks = tracker.update(frame, detections, reid)

        elapsed = time.time() - start
        total_time += elapsed

        fps_now = 1 / elapsed if elapsed > 0 else 0

        cv2.putText(frame, f"FPS: {fps_now:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        frame = draw_tracks(frame, tracks)

        for t in tracks:
            mot_writer.write(i, t["id"], t["bbox"])

        writer.write(frame)

        if i % 50 == 0:
            print(f"Frame {i}/{len(frames)} FPS={fps_now:.2f}")

    writer.release()
    mot_writer.close()

    avg_fps = len(frames) / total_time if total_time > 0 else 0

    print("DONE")
    print("AVG FPS:", avg_fps)

    return avg_fps

detectors = [
    "yolov5n",
    "yolov5m",
    "yolov5l6",
    "detectron2"
]


reids = [
    "osnet_x0_25",
    "osnet_x1_0",
    "resnet50"
]


confidences = [
    # 0.3,
    0.5,
    # 0.7
]


cosine_distances = [
#     0.2,
#     0.3,
    0.5
]


nn_budgets = [
    50,
    # 100
]
