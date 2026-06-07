import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage1.train_runner import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Stage 1 training/evaluation pass.")
    parser.add_argument("--config", required=True, help="Path to a JSON-compatible YAML config file.")
    args = parser.parse_args()
    metrics = run_training(args.config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
