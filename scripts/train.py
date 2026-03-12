"""Launch training."""
import yaml
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training.trainer import BubbleTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    trainer = BubbleTrainer(config, device="cuda")
    trainer.train()


if __name__ == "__main__":
    main()
