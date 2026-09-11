# DeepSORT Multi-Object Tracking

Проект по **Multi-Object Tracking** на основе DeepSORT.

## Цель

Исследовать влияние модели детекции и Person Re-Identification на качество и скорость трекинга.

## Pipeline

`Video → Detector → Person ReID → DeepSORT → MOT results → TrackEval`

## Компоненты

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

Для выбора итоговых конфигураций применяется ограничение `FPS >= 5`.

## Результаты исходного ноутбука

После фильтрации `FPS >= 5` в исходном ноутбуке выбиралась конфигурация с максимальным HOTA для каждой последовательности.

| Sequence | Detector | ReID | FPS | MOTA | IDF1 | HOTA |
|---|---|---|---:|---:|---:|---:|
| KITTI-17 | YOLOv5m | ResNet50 | 22.21 | 66.33 | 80.77 | 58.47 |
| MOT16-09 | YOLOv5l6 | OSNet x0.25 | 6.51 | 45.25 | 51.28 | 42.42 |
| MOT16-11 | YOLOv5l6 | OSNet x1.0 | 6.58 | 52.31 | 56.85 | 50.49 |
| PETS09-S2L1 | YOLOv5l6 | OSNet x1.0 | 6.38 | 82.51 | 75.86 | 57.75 |
| TUD-Campus | YOLOv5l6 | OSNet x0.25 | 5.79 | 60.17 | 65.21 | 48.71 |
| TUD-Stadtmitte | YOLOv5n | OSNet x0.25 | 17.66 | 77.85 | 71.36 | 56.17 |

Эти значения перенесены из сохранённых результатов исходного ноутбука; после модульного разбиения код рекомендуется повторно прогнать для проверки воспроизводимости.



## Запуск

```bash
python main.py
```

Для Google Colab необходимо сначала настроить окружение и путь к Google Drive в `ProjectConfig.base_dir`.

## Исходные материалы

Исходный ноутбук содержит реализацию детекторов, ReID, DeepSORT, сохранение MOT-результатов, TrackEval и grid search.
