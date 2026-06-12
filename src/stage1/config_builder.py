import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class BackboneSpec:
    slug: str
    model_name: str
    local_model_dir: str
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    backend_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


LARGE_BACKBONE_CONFIG_OVERRIDES = {
    "model_dtype": "bfloat16",
    "learning_rate": 0.00002,
}


LARGE_BACKBONE_BACKEND_OVERRIDES = {
    "hf": {
        "batch_size": 2,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
    },
    "hf_rsg": {
        "batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
    },
}


SCIFIVE_BASE_PUBMED_PMC = BackboneSpec(
    slug="scifive",
    model_name="razent/SciFive-base-Pubmed_PMC",
    local_model_dir="models/stage1/scifive-base-pubmed-pmc",
)


SCIFIVE_LARGE_PUBMED_PMC = BackboneSpec(
    slug="scifive_large",
    model_name="razent/SciFive-large-Pubmed_PMC",
    local_model_dir="models/stage1/scifive-large-pubmed-pmc",
    config_overrides=LARGE_BACKBONE_CONFIG_OVERRIDES,
    backend_overrides=LARGE_BACKBONE_BACKEND_OVERRIDES,
)


BIOBART_V2_BASE = BackboneSpec(
    slug="biobart_v2_base",
    model_name="GanjinZero/biobart-v2-base",
    local_model_dir="models/stage1/biobart-v2-base",
)


BIOBART_V2_LARGE = BackboneSpec(
    slug="biobart_v2_large",
    model_name="GanjinZero/biobart-v2-large",
    local_model_dir="models/stage1/biobart-v2-large",
    config_overrides=LARGE_BACKBONE_CONFIG_OVERRIDES,
    backend_overrides=LARGE_BACKBONE_BACKEND_OVERRIDES,
)


CLINICALT5_BASE = BackboneSpec(
    slug="clinicalt5_base",
    model_name="luqh/ClinicalT5-base",
    local_model_dir="models/stage1/clinicalt5-base",
)


BACKBONES = {
    "biobart-v2-base": BIOBART_V2_BASE,
    "biobart-v2-large": BIOBART_V2_LARGE,
    "clinicalt5-base": CLINICALT5_BASE,
    "scifive-base": SCIFIVE_BASE_PUBMED_PMC,
    "scifive-large": SCIFIVE_LARGE_PUBMED_PMC,
}


def make_custom_backbone_spec(
    slug: str,
    model_name: str,
    local_model_dir: str,
    size_profile: str = "base",
) -> BackboneSpec:
    normalized_profile = str(size_profile).lower()
    if normalized_profile == "base":
        return BackboneSpec(slug=slug, model_name=model_name, local_model_dir=local_model_dir)
    if normalized_profile == "large":
        return BackboneSpec(
            slug=slug,
            model_name=model_name,
            local_model_dir=local_model_dir,
            config_overrides=LARGE_BACKBONE_CONFIG_OVERRIDES,
            backend_overrides=LARGE_BACKBONE_BACKEND_OVERRIDES,
        )
    raise ValueError(f"Unknown size_profile: {size_profile}")


def read_stage1_config(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_stage1_config(path: str, config: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def derive_backbone_config(base_config: Dict[str, Any], backbone: BackboneSpec) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["model"] = backbone.model_name
    config["model_name_or_path"] = backbone.local_model_dir
    config["experiment_id"] = _derive_experiment_id(str(config["experiment_id"]), backbone.slug)
    config["output_dir"] = _derive_output_dir(str(config["output_dir"]), backbone.slug, str(config.get("backend", "")))
    config.update(backbone.config_overrides)
    config.update(backbone.backend_overrides.get(str(config.get("backend", "")), {}))
    return config


def derive_config_filename(base_config_path: str, backbone: BackboneSpec) -> str:
    name = Path(base_config_path).name
    if "_biobart_" in name:
        return name.replace("_biobart_", f"_{backbone.slug}_")
    if "_hf_rsg_" in name:
        return name.replace("_hf_rsg_", f"_hf_rsg_{backbone.slug}_", 1)
    return name


def derive_many_backbone_configs(base_config_paths: Iterable[str], backbone: BackboneSpec) -> List[Dict[str, Any]]:
    return [derive_backbone_config(read_stage1_config(path), backbone) for path in base_config_paths]


def _derive_experiment_id(experiment_id: str, slug: str) -> str:
    if "_biobart_" in experiment_id:
        return experiment_id.replace("_biobart_", f"_{slug}_")
    if "_hf_rsg_" in experiment_id:
        return experiment_id.replace("_hf_rsg_", f"_hf_rsg_{slug}_", 1)
    return experiment_id


def _derive_output_dir(output_dir: str, slug: str, backend: str) -> str:
    normalized = output_dir.replace("\\", "/")
    if "biobart_text2text" in normalized:
        return normalized.replace("biobart_text2text", f"{slug}_text2text", 1)
    if backend == "hf_rsg" and "rsg_biore" in normalized:
        return normalized.replace("rsg_biore", f"rsg_biore_{slug}", 1)
    return normalized
