import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class BackboneSpec:
    slug: str
    model_name: str
    local_model_dir: str


SCIFIVE_BASE_PUBMED_PMC = BackboneSpec(
    slug="scifive",
    model_name="razent/SciFive-base-Pubmed_PMC",
    local_model_dir="models/stage1/scifive-base-pubmed-pmc",
)


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
