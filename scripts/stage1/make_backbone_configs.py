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
    make_custom_backbone_spec,
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
    parser = argparse.ArgumentParser(
        description="Create ChemProt Stage 1 configs for an HF text-to-text backbone."
    )
    parser.add_argument("--output-dir", default="configs/stage1")
    parser.add_argument(
        "--backbone",
        choices=sorted(BACKBONES),
        default="scifive-base",
        help="Built-in backbone key. Ignored when all custom backbone arguments are provided.",
    )
    parser.add_argument("--custom-slug", help="Custom backbone slug used in experiment ids and output dirs.")
    parser.add_argument("--custom-model", help="HF model id recorded in the config metadata.")
    parser.add_argument("--custom-model-dir", help="Local model directory used by transformers.from_pretrained().")
    parser.add_argument(
        "--size-profile",
        choices=["base", "large"],
        default="base",
        help="Training override profile for custom backbones.",
    )
    parser.add_argument("--base-config", action="append", dest="base_configs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backbone = resolve_backbone(args)
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

    print(
        json.dumps(
            {
                "backbone": backbone.slug,
                "model": backbone.model_name,
                "model_name_or_path": backbone.local_model_dir,
                "configs": written,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def resolve_backbone(args: argparse.Namespace):
    custom_values = [args.custom_slug, args.custom_model, args.custom_model_dir]
    if any(custom_values):
        if not all(custom_values):
            raise SystemExit("--custom-slug, --custom-model, and --custom-model-dir must be provided together.")
        return make_custom_backbone_spec(
            slug=args.custom_slug,
            model_name=args.custom_model,
            local_model_dir=args.custom_model_dir,
            size_profile=args.size_profile,
        )
    return BACKBONES[args.backbone]


if __name__ == "__main__":
    main()
