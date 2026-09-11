"""TrackEval evaluation utilities."""

import os
import shutil
import tempfile
import numpy as np

from TrackEval.trackeval import Evaluator
from TrackEval.trackeval.datasets import MotChallenge2DBox
from TrackEval.trackeval.metrics import HOTA, CLEAR, Identity


def compute_metrics(sequence_path, pred_file):
    print(f"Evaluating sequence: {sequence_path}")
    seq_name = os.path.basename(sequence_path)

    tmp = tempfile.mkdtemp()
    gt_root = os.path.join(tmp, "gt")
    tracker_root = os.path.join(tmp, "trackers")


    # ВАЖНО: Мы жестко фиксируем BENCHMARK = "MOT16" для ВСЕХ папок. так как все лежат в одной папке на подобии MOT-16
    # Это заставит TrackEval использовать один и тот же предсказуемый шаблон путей.
    benchmark = "MOT16"

    # Создаем правильную структуру папок, которую ожидает MOT16 при SKIP_SPLIT_FOL=True
    gt_seq_dir = os.path.join(gt_root, seq_name)
    os.makedirs(gt_seq_dir, exist_ok=True)

    # Копируем файл конфигурации seqinfo.ini
    src_ini = os.path.join(sequence_path, "seqinfo.ini")
    dst_ini = os.path.join(gt_seq_dir, "seqinfo.ini")
    shutil.copy(src_ini, dst_ini)

    # Создаём папку gt и копируем туда gt.txt
    gt_gt_dir = os.path.join(gt_seq_dir, "gt")
    os.makedirs(gt_gt_dir, exist_ok=True)
    shutil.copy(
        os.path.join(sequence_path, "gt", "gt.txt"),
        os.path.join(gt_gt_dir, "gt.txt")
    )

    # Результаты трекера складываем по структуре, которую TrackEval/MOT16 считает стандартной
    tracker_name = "DeepSORT"

    # Для MOT16 + SKIP_SPLIT_FOL=True библиотека ищет файл в: {TRACKERS_FOLDER}/{tracker_name}/data/{seq_name}.txt
    tracker_dir = os.path.join(tracker_root, tracker_name, "data")
    os.makedirs(tracker_dir, exist_ok=True)
    shutil.copy(pred_file, os.path.join(tracker_dir, f"{seq_name}.txt"))

    # Генерируем seqmap-файл
    seqmap_dir = os.path.join(tmp, "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    seqmap_path = os.path.join(seqmap_dir, f"{benchmark}-train.txt")

    with open(seqmap_path, "w") as f:
        f.write("name\n")
        f.write(seq_name + "\n")

    # Настройки самого оценщика
    eval_config = Evaluator.get_default_eval_config()
    dataset_config = MotChallenge2DBox.get_default_dataset_config()

    dataset_config["GT_FOLDER"] = gt_root
    dataset_config["TRACKERS_FOLDER"] = tracker_root
    dataset_config["BENCHMARK"] = benchmark
    dataset_config["SKIP_SPLIT_FOL"] = True  # Игнорируем деление на train/test подпапки
    dataset_config["SEQMAP_FOLDER"] = seqmap_dir
    dataset_config["TRACKERS_TO_EVAL"] = [tracker_name]
    dataset_config["DO_PREPROC"] = False
    # Дополнительно настраиваем маску путей для GT, так как у нас плоская структура
    dataset_config["GT_LOC_FORMAT"] = "{gt_folder}/{seq}/gt/gt.txt"

    evaluator = Evaluator(eval_config)
    dataset = MotChallenge2DBox(dataset_config)

    metrics = [HOTA(), CLEAR(), Identity()]
    results, _ = evaluator.evaluate([dataset], metrics)

    res_data = results["MotChallenge2DBox"][tracker_name]["COMBINED_SEQ"]

    hota_val = res_data["HOTA"]["HOTA"]
    if isinstance(hota_val, np.ndarray) or isinstance(hota_val, list):
        hota = float(np.mean(hota_val)) # переводим в проценты, если нужно
    else:
        hota = float(hota_val) if float(hota_val) <= 1.0 else float(hota_val)

    mota = float(res_data["CLEAR"]["MOTA"])
    idf1 = float(res_data["Identity"]["IDF1"])

    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    return {
        "MOTA": mota if mota is not None else 0.0,
        "IDF1": idf1 if idf1 is not None else 0.0,
        "HOTA": hota if hota is not None else 0.0,
    }
