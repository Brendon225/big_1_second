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


def build_marked_relation_prompt(sample: Dict[str, str], schema: RelationSchema, semantic_field: str) -> str:
    options = build_relation_options(schema, semantic_field)
    marked_text = mark_entity_pair_text(sample["text"], sample["head_entity"], sample["tail_entity"])
    return (
        "Task: identify the biomedical relation for the marked entity pair.\n"
        f"Text: {marked_text}\n"
        f"Head entity: {sample['head_entity']} ({sample['head_type']})\n"
        f"Tail entity: {sample['tail_entity']} ({sample['tail_type']})\n"
        "Relation schema:\n"
        f"{options}\n"
        "Answer with exactly this format: relation: <label>"
    )


def mark_entity_pair_text(text: str, head_entity: str, tail_entity: str) -> str:
    marked = replace_first(text, head_entity, f"<H> {head_entity} </H>")
    marked = replace_first(marked, tail_entity, f"<T> {tail_entity} </T>")
    return marked


def replace_first(text: str, needle: str, replacement: str) -> str:
    if not needle:
        return text
    index = text.find(needle)
    if index < 0:
        return text
    return text[:index] + replacement + text[index + len(needle) :]


def build_target_text(sample: Dict[str, str]) -> str:
    return f"relation: {sample['gold_relation']}"
