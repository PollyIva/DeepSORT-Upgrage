"""Central project configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectConfig:
    base_dir: Path = Path(__file__).resolve().parent.parent
    sequences: list[str] = field(default_factory=lambda: [
        "KITTI-17", "MOT16-09", "MOT16-11",
        "PETS09-S2L1", "TUD-Campus", "TUD-Stadtmitte"
    ])
    detectors: list[str] = field(default_factory=lambda: [
        "yolov5n", "yolov5m", "yolov5l6", "detectron2"
    ])
    reid_models: list[str] = field(default_factory=lambda: [
        "osnet_x0_25", "osnet_x1_0", "resnet50"
    ])
    confidences: list[float] = field(default_factory=lambda: [0.5])
    cosine_distances: list[float] = field(default_factory=lambda: [0.5])
    nn_budgets: list[int] = field(default_factory=lambda: [50])
    min_fps: float = 5.0

    @property
    def data_dir(self):
        return self.base_dir / "MOT_datasets"

    @property
    def videos_dir(self):
        return self.base_dir / "videos"

    @property
    def results_dir(self):
        return self.base_dir / "results"

    @property
    def metrics_dir(self):
        return self.base_dir / "metrics"

    @property
    def cache_dir(self):
        return self.base_dir / "cache"

    def create_dirs(self):
        for path in (
            self.base_dir, self.videos_dir, self.results_dir,
            self.metrics_dir, self.cache_dir
        ):
            path.mkdir(parents=True, exist_ok=True)
