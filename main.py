from src.config import ProjectConfig
from src.experiment import run_grid


if __name__ == "__main__":
    config = ProjectConfig()
    config.create_dirs()
    output = run_grid(config)
    print(f"Results saved to: {output}")
