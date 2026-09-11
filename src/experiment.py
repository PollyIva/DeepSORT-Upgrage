"""Experiment execution and grid search."""

import csv
from pathlib import Path

from .detectors import create_detector
from .reid import create_reid
from .tracker import DeepSORTWrapper
from .pipeline import run_sequence
from .evaluation import compute_metrics
from .utils import make_key, load_cache, save_cache


def run_experiment(video_path, sequence, detector_name, reid_name,
                   conf, cosine_distance, nn_budget, base_dir):
    sequence_name = Path(sequence).name

    detector = create_detector(detector_name, conf=conf)
    reid = create_reid(reid_name)

    tracker = DeepSORTWrapper(
        max_cosine_distance=cosine_distance,
        nn_budget=nn_budget
    )

    videos_dir = Path(base_dir) / "videos"
    results_dir = Path(base_dir) / "results"
    videos_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    output_video = videos_dir / f"{sequence_name}_{detector_name}_{reid_name}.mp4"
    output_mot = results_dir / f"{sequence_name}_{detector_name}_{reid_name}.txt"

    fps = run_sequence(
        img_dir=str(video_path),
        output_video=str(output_video),
        mot_path=str(output_mot),
        detector=detector,
        reid=reid,
        tracker=tracker
    )

    return {"fps": fps, "mot_file": str(output_mot)}


def run_cached(sequence, video_path, detector, reid,
               conf, cosine, budget, config):
    key = make_key(sequence, detector, reid, conf, cosine, budget)
    cached = load_cache(config.cache_dir, key)

    if cached:
        return cached

    result = run_experiment(
        video_path, sequence, detector, reid,
        conf, cosine, budget, config.base_dir
    )
    save_cache(config.cache_dir, key, result)
    return result


def run_grid(config):
    output_csv = config.metrics_dir / "experiments.csv"
    fieldnames = [
        "sequence", "detector", "reid", "conf",
        "cosine_distance", "nn_budget", "fps", "MOTA", "IDF1", "HOTA"
    ]

    processed = set()

    if output_csv.exists():
        with output_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                processed.add((
                    row["sequence"], row["detector"], row["reid"],
                    float(row["conf"]), float(row["cosine_distance"]),
                    int(row["nn_budget"])
                ))

    file_exists = output_csv.exists()

    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for sequence in config.sequences:
            sequence_path = config.data_dir / sequence
            video_path = sequence_path / "img1"

            for detector in config.detectors:
                for reid in config.reid_models:
                    for conf in config.confidences:
                        for cosine in config.cosine_distances:
                            for budget in config.nn_budgets:
                                key = (
                                    sequence, detector, reid,
                                    float(conf), float(cosine), int(budget)
                                )

                                if key in processed:
                                    continue

                                exp = run_cached(
                                    sequence_path, video_path,
                                    detector, reid, conf, cosine, budget, config
                                )

                                metrics = compute_metrics(
                                    str(sequence_path), exp["mot_file"]
                                )

                                row = {
                                    "sequence": sequence,
                                    "detector": detector,
                                    "reid": reid,
                                    "conf": conf,
                                    "cosine_distance": cosine,
                                    "nn_budget": budget,
                                    "fps": exp["fps"],
                                    **metrics,
                                }

                                writer.writerow(row)
                                f.flush()
                                processed.add(key)

    return output_csv
