from typing import Any

from src.stage1.models.rsg_biore import RsgBioREModel
from src.stage1.models.text2text_baseline import Text2TextBaseline
from src.stage1.optional_deps import require_modules
from src.stage1.schema import RelationSchema


def build_stage1_model(
    method: str,
    schema: RelationSchema,
    semantic_field: str,
    backend: str = "mock",
    **kwargs: Any,
) -> Any:
    if backend == "mock":
        return _build_mock_model(method=method, schema=schema, semantic_field=semantic_field, **kwargs)
    if backend == "fake_train":
        return _build_mock_model(method=method, schema=schema, semantic_field=semantic_field, **kwargs)
    if backend in {"hf", "hf_rsg"}:
        require_modules(["torch", "transformers"], "Hugging Face T5/BioBART training")
        backend_class = resolve_hf_backend_class(backend)

        init_kwargs = {
            "schema": schema,
            "semantic_field": semantic_field,
            "model_name_or_path": kwargs.get("model_name_or_path", "t5-small"),
            "max_input_length": int(kwargs.get("max_input_length", 512)),
            "max_output_length": int(kwargs.get("max_output_length", 32)),
            "device": kwargs.get("device"),
            "model_dtype": kwargs.get("model_dtype", "float32"),
            "decoding_strategy": kwargs.get("decoding_strategy", "generate"),
            "tokenizer_load_kwargs": kwargs.get("tokenizer_load_kwargs"),
            "model_load_kwargs": kwargs.get("model_load_kwargs"),
        }
        if backend == "hf_rsg":
            init_kwargs.update(
                {
                    "alignment_lambda": float(kwargs.get("alignment_lambda", 0.1)),
                    "temperature_tau": float(kwargs.get("temperature_tau", 0.1)),
                    "prototype_type": kwargs.get("prototype_type", "learnable"),
                    "prototype_semantic_field": kwargs.get("prototype_semantic_field"),
                    "instance_pooling": kwargs.get("instance_pooling", "mean"),
                    "use_entity_markers": bool(kwargs.get("use_entity_markers", False)),
                    "use_prototype_fusion": bool(kwargs.get("use_prototype_fusion", False)),
                    "prototype_fusion_alpha": float(kwargs.get("prototype_fusion_alpha", 0.0)),
                }
            )
        return backend_class(**init_kwargs)
    raise ValueError(f"Unknown backend: {backend}")


def resolve_hf_backend_class(backend: str) -> Any:
    if backend == "hf":
        from src.stage1.hf_text2text_backend import HfText2TextModel

        return HfText2TextModel
    if backend == "hf_rsg":
        from src.stage1.hf_rsg_biore_backend import HfRsgBioREModel

        return HfRsgBioREModel
    raise ValueError(f"Unknown Hugging Face backend: {backend}")


def _build_mock_model(method: str, schema: RelationSchema, semantic_field: str, **kwargs: Any) -> Any:
    if method.startswith("R"):
        return RsgBioREModel(
            schema=schema,
            semantic_field=semantic_field,
            alignment_lambda=float(kwargs.get("alignment_lambda", 0.1)),
            temperature_tau=float(kwargs.get("temperature_tau", 0.1)),
            prototype_type=kwargs.get("prototype_type", "learnable"),
            backend="mock",
        )
    return Text2TextBaseline(schema=schema, semantic_field=semantic_field, backend="mock")
