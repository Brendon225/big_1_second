import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


REQUIRED_FIELDS = [
    "raw_label",
    "label_words",
    "relation_description",
    "entity_type_aware_description",
    "knowledge_enhanced_description",
]


@dataclass(frozen=True)
class RelationEntry:
    raw_label: str
    label_words: str
    relation_description: str
    entity_type_aware_description: str
    knowledge_enhanced_description: str
    head_type: str = ""
    tail_type: str = ""
    is_negative: bool = False

    def semantic_text(self, field: str) -> str:
        if field not in REQUIRED_FIELDS:
            raise ValueError(f"Unknown semantic field: {field}")
        return str(getattr(self, field))


@dataclass(frozen=True)
class RelationSchema:
    dataset: str
    relations: List[RelationEntry]
    rare_relations: List[str]
    no_relation_labels: List[str]

    @property
    def labels(self) -> List[str]:
        return [entry.raw_label for entry in self.relations]

    @property
    def positive_labels(self) -> List[str]:
        blocked = set(self.no_relation_labels)
        return [label for label in self.labels if label not in blocked]

    def by_label(self) -> Dict[str, RelationEntry]:
        return {entry.raw_label: entry for entry in self.relations}

    def semantic_items(self, field: str) -> Iterable[tuple[str, str]]:
        for entry in self.relations:
            yield entry.raw_label, entry.semantic_text(field)


def load_relation_schema(path: str) -> RelationSchema:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} must be JSON-compatible YAML for the stdlib smoke runner. "
            "Install PyYAML later if you want general YAML syntax."
        ) from exc

    relations = []
    for index, item in enumerate(payload.get("relations", [])):
        missing = [field for field in REQUIRED_FIELDS if field not in item]
        if missing:
            raise ValueError(f"Relation schema item {index} misses fields: {missing}")
        relations.append(
            RelationEntry(
                raw_label=item["raw_label"],
                label_words=item["label_words"],
                relation_description=item["relation_description"],
                entity_type_aware_description=item["entity_type_aware_description"],
                knowledge_enhanced_description=item["knowledge_enhanced_description"],
                head_type=item.get("head_type", ""),
                tail_type=item.get("tail_type", ""),
                is_negative=bool(item.get("is_negative", False)),
            )
        )

    if not relations:
        raise ValueError(f"No relations found in schema: {path}")

    return RelationSchema(
        dataset=payload.get("dataset", ""),
        relations=relations,
        rare_relations=list(payload.get("rare_relations", [])),
        no_relation_labels=list(payload.get("no_relation_labels", [])),
    )
