# DeepSORT Multi-Object Tracking

A **Multi-Object Tracking** project based on DeepSORT.

## Goal

Investigate how the detection model and Person Re-Identification affect tracking quality and speed.

## Pipeline

`Video → Detector → Person ReID → DeepSORT → MOT results → TrackEval`

## Components

**Detectors**
- YOLOv5n
- YOLOv5m
- YOLOv5l6
- Detectron2 / Mask R-CNN R50-FPN

**ReID**
- OSNet x0.25
- OSNet x1.0
- ResNet50

**DeepSORT parameters**
- confidence threshold
- maximum cosine distance
- NN budget

## Results
Best Videos: https://drive.google.com/drive/folders/1iYmanHW2JNYxTIGtqarRTaJLcHskqI2Q

All Videos: https://drive.google.com/drive/folders/1A81lj6PQ3d9AP3m4yS6rru7M6FwnuaJ4

Metrics: https://github.com/PollyIva/DL_in_CV/blob/main/experiments.csv

## Dataset

- KITTI-17
- MOT16-09
- MOT16-11
- PETS09-S2L1
- TUD-Campus
- TUD-Stadtmitte

## Metrics

- FPS
- MOTA
- IDF1
- HOTA

The `FPS >= 5` constraint is used to select the final configurations.

## Results from the original notebook

After filtering by `FPS >= 5`, the configuration with the maximum HOTA was selected for each sequence in the original notebook.

| Sequence | Detector | ReID | FPS | MOTA | IDF1 | HOTA |
|---|---|---:|---:|---:|---:|---:|
| KITTI-17 | YOLOv5m | ResNet50 | 22.21 | 66.33 | 80.77 | 58.47 |
| MOT16-09 | YOLOv5l6 | OSNet x0.25 | 6.51 | 45.25 | 51.28 | 42.42 |
| MOT16-11 | YOLOv5l6 | OSNet x1.0 | 6.58 | 52.31 | 56.85 | 50.49 |
| PETS09-S2L1 | YOLOv5l6 | OSNet x1.0 | 6.38 | 82.51 | 75.86 | 57.75 |
| TUD-Campus | YOLOv5l6 | OSNet x0.25 | 5.79 | 60.17 | 65.21 | 48.71 |
| TUD-Stadtmitte | YOLOv5n | OSNet x0.25 | 17.66 | 77.85 | 71.36 | 56.17 |

These values were taken from the saved results of the original notebook; after modularizing the code, it is recommended to re-run the pipeline to verify reproducibility.

## Running

```bash
python main.py
```

For Google Colab, first configure the environment and the path to Google Drive in `ProjectConfig.base_dir`.

## Source materials

The original notebook contains implementations of detectors, ReID, DeepSORT, MOT result saving, TrackEval, and grid search.