import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage1.chemprot_converter import convert_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ChemProt raw TSV files into unified Stage 1 JSONL."
        )
    )
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--max-negative-per-doc", type=int, default=2)
    parser.add_argument("--no-negatives", action="store_true")
    args = parser.parse_args()
    counts = convert_corpus(
        corpus_root=args.raw_dir,
        output_dir=args.output_dir,
        include_negatives=not args.no_negatives,
        max_negative_per_doc=args.max_negative_per_doc,
        max_samples_per_split=args.max_samples_per_split,
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
