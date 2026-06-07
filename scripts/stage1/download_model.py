import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage1.optional_deps import require_modules


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face seq2seq model for Stage 1.")
    parser.add_argument("--model-name", default="GanjinZero/biobart-base")
    parser.add_argument("--output-dir", default="models/stage1/biobart-base")
    args = parser.parse_args()

    require_modules(["torch", "transformers"], "model download")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer.save_pretrained(target)
    model.save_pretrained(target)
    manifest = {
        "source_model": args.model_name,
        "local_dir": str(target),
        "files": sorted(path.name for path in target.iterdir() if path.is_file()),
    }
    (target / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
