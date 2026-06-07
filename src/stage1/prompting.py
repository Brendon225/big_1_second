from typing import Dict

from src.stage1.schema import RelationSchema


def build_relation_options(schema: RelationSchema, semantic_field: str) -> str:
    lines = []
    for label, text in schema.semantic_items(semantic_field):
        lines.append(f"- {label}: {text}")
    return "\n".join(lines)


def build_relation_prompt(sample: Dict[str, str], schema: RelationSchema, semantic_field: str) -> str:
    options = build_relation_options(schema, semantic_field)
    return (
        "Task: identify the biomedical relation for the given entity pair.\n"
        f"Text: {sample['text']}\n"
        f"Head entity: {sample['head_entity']} ({sample['head_type']})\n"
        f"Tail entity: {sample['tail_entity']} ({sample['tail_type']})\n"
        "Relation schema:\n"
        f"{options}\n"
        "Answer with exactly this format: relation: <label>"
    )


def build_target_text(sample: Dict[str, str]) -> str:
    return f"relation: {sample['gold_relation']}"
