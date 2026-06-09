import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage1.result_summary import summarize_metrics, write_summary_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage 1 metrics.json files into one CSV table.")
    parser.add_argument("--root-dir", default="outputs/stage1/chemprot", help="Directory containing Stage 1 run outputs.")
    parser.add_argument("--output", default="outputs/stage1/chemprot/stage1_metrics_summary.csv")
    args = parser.parse_args()

    rows = summarize_metrics(args.root_dir)
    write_summary_csv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
