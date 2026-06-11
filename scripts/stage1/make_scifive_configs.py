import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage1.config_builder import (
    BACKBONES,
    derive_backbone_config,
    derive_config_filename,
    read_stage1_config,
    write_stage1_config,
)


DEFAULT_BASE_CONFIGS = [
    "configs/stage1/chemprot_hf_biobart_P1_full_seed42.yaml",
    "configs/stage1/chemprot_hf_biobart_P2_full_seed42.yaml",
    "configs/stage1/chemprot_hf_biobart_P3_full_seed42.yaml",
    "configs/stage1/chemprot_hf_rsg_R2_1_full_seed42.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ChemProt Stage 1 backbone configs from existing BioBART configs.")
    parser.add_argument("--output-dir", default="configs/stage1")
    parser.add_argument(
        "--backbone",
        choices=sorted(BACKBONES),
        default="scifive-base",
        help="Backbone variant to generate. Default preserves the original SciFive-base behavior.",
    )
    parser.add_argument("--base-config", action="append", dest="base_configs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backbone = BACKBONES[args.backbone]
    base_configs = args.base_configs or DEFAULT_BASE_CONFIGS
    output_dir = Path(args.output_dir)
    written = []

    for base_path in base_configs:
        base_config = read_stage1_config(base_path)
        derived = derive_backbone_config(base_config, backbone)
        target = output_dir / derive_config_filename(base_path, backbone)
        written.append(str(target))
        if not args.dry_run:
            write_stage1_config(str(target), derived)

    print(json.dumps({"backbone": args.backbone, "configs": written, "dry_run": args.dry_run}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
