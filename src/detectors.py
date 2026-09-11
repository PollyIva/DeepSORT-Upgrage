"""Object detector implementations."""

import numpy as np
from ultralytics import YOLO

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
except ImportError:
    DefaultPredictor = get_cfg = model_zoo = None


class Detector:
    def detect(self, frame):
        raise NotImplementedError


class YOLOv5Detector(Detector):
    def __init__(self, model_name="yolov5n.pt", conf=0.5, device=DEVICE):
        self.model = YOLO(model_name)  # ultralytics wrapper
        self.conf = conf
        self.device = device

    def detect(self, frame):
        result = self.model(frame, conf=self.conf, device=self.device, verbose=False)[0]

        detections = []
        for box in result.boxes:
            if int(box.cls[0]) != 0:  # person class
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            detections.append([
                int(x1), int(y1), int(x2), int(y2), float(box.conf[0])
            ])

        return detections

# !pip install 'git+https://github.com/facebookresearch/detectron2.git'

# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2 import model_zoo


class Detectron2Detector(Detector):

    def __init__(self,
                 conf=0.5,
                 device=DEVICE,
                 max_retries=3):

        import time

        cfg = get_cfg()

        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf
        cfg.MODEL.DEVICE = device

        for attempt in range(max_retries):
            try:
                self.predictor = DefaultPredictor(cfg)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    raise
    def detect(self, frame):

        outputs = self.predictor(frame)

        instances = outputs["instances"].to("cpu")

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        detections = []

        for box, score, cls in zip(boxes, scores, classes):

            # person = 0 в COCO
            if cls != 0:
                continue

            x1, y1, x2, y2 = box

            detections.append([
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                float(score)
            ])

        return detections

#  Фабрика детекторов

def create_detector(name, conf=0.5, device=DEVICE):
    if name == "yolov5n":
        return YOLOv5Detector("yolov5n.pt", conf=conf, device=device)
    if name == "yolov5m":
        return YOLOv5Detector("yolov5m.pt", conf=conf, device=device)
    if name == "yolov5l6":
        return YOLOv5Detector("yolov5l6.pt", conf=conf, device=device)
    if name == "detectron2":
        return Detectron2Detector(conf=conf, device=device)
    raise ValueError(f"Unknown detector: {name}")
