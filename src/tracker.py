"""DeepSORT tracking wrapper."""

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


class DeepSORTWrapper:
    def __init__(self, max_cosine_distance=0.3, nn_budget=100):
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine",
            max_cosine_distance,
            nn_budget
        )
        self.tracker = Tracker(self.metric)

    def update(self, frame, detections, reid):
        """
        detections: [[x1, y1, x2, y2, confidence], ...]
        """
        valid_detections = []
        crops = []

        # Шаг 1: Только собираем кропы и фильтруем пустые детекции
        for det in detections:
            x1, y1, x2, y2, conf = det

            # Ограничиваем координаты границами кадра, чтобы избежать пустых или искаженных кропов
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crops.append(crop)
            valid_detections.append([x1, y1, x2, y2, conf])

        # Шаг 2: Извлекаем эмбеддинги ОДНИМ батчем (вызов GPU происходит один раз)
        features = reid.get_embeddings_batch(crops)

        # Шаг 3: Формируем объекты Detection для DeepSORT
        ds_detections = []
        for det, feature in zip(valid_detections, features):
            x1, y1, x2, y2, conf = det

            # Переводим в формат tlwh (top_left_x, top_left_y, width, height)
            bbox = [x1, y1, x2 - x1, y2 - y1]

            ds_detections.append(
                Detection(bbox, conf, feature)
            )

        # Шаг 4: Стандартный пайплайн трекера
        self.tracker.predict()
        self.tracker.update(ds_detections)

        results = []
        for track in self.tracker.tracks:
            if not track.is_confirmed():
                continue
            if track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            results.append({
                "id": track.track_id,
                "bbox": bbox
            })

        return results

tracker = DeepSORTWrapper(
    max_cosine_distance=0.3,
    nn_budget=100
)
detector = create_detector("yolov5n")
reid = create_reid("osnet_x0_25")
