import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = "https://huggingface.co/datasets/bigbio/chemprot/resolve/main/ChemProt_Corpus.zip"
DEFAULT_SHA256 = "492e3d607f38e2727b799e9d60263b776ebd2a5e61cf0fb59bea2b3eb68e1c28"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract ChemProt raw corpus zip.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", default="data/stage1/raw/chemprot")
    parser.add_argument("--sha256", default=DEFAULT_SHA256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "ChemProt_Corpus.zip"
    extracted_dir = output_dir / "extracted"

    if args.force or not zip_path.exists():
        download_file(args.url, zip_path)

    actual_sha = sha256_file(zip_path)
    if args.sha256 and actual_sha != args.sha256:
        raise SystemExit(f"SHA256 mismatch for {zip_path}: expected {args.sha256}, got {actual_sha}")

    extracted_dir.mkdir(parents=True, exist_ok=True)
    extract_zip(zip_path, extracted_dir)
    extract_nested_zips(extracted_dir)

    manifest = {
        "source_url": args.url,
        "zip_path": str(zip_path),
        "sha256": actual_sha,
        "bytes": zip_path.stat().st_size,
        "extracted_dir": str(extracted_dir),
        "files": sorted(str(path.relative_to(extracted_dir)) for path in extracted_dir.rglob("*") if path.is_file()),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def download_file(url: str, output_path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "rsg-biore-stage1/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        output_path.write_bytes(response.read())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def extract_nested_zips(root: Path) -> None:
    for zip_path in sorted(root.rglob("*.zip")):
        nested_dir = zip_path.with_suffix("")
        nested_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(nested_dir)


if __name__ == "__main__":
    main()
