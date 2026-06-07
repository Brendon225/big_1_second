from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


POSITIVE_LABELS = {"CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9"}


@dataclass(frozen=True)
class ChemProtEntity:
    pmid: str
    entity_id: str
    raw_type: str
    start: str
    end: str
    mention: str

    @property
    def normalized_type(self) -> str:
        raw = self.raw_type.upper()
        if raw.startswith("CHEMICAL"):
            return "chemical"
        if raw.startswith("GENE"):
            return "protein"
        return raw.lower()


def convert_corpus(
    corpus_root: str,
    output_dir: str,
    include_negatives: bool = True,
    max_negative_per_doc: int = 2,
    max_samples_per_split: Optional[int] = None,
) -> Dict[str, int]:
    root = Path(corpus_root)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for split_name, raw_name in {
        "train": "chemprot_training",
        "dev": "chemprot_development",
        "test": "chemprot_test_gs",
    }.items():
        split_dir = find_split_dir(root, raw_name)
        samples = convert_split(
            split_dir=split_dir,
            split_name=split_name,
            include_negatives=include_negatives,
            max_negative_per_doc=max_negative_per_doc,
        )
        if max_samples_per_split is not None:
            samples = samples[:max_samples_per_split]
        write_jsonl(target / f"{split_name}.jsonl", samples)
        counts[split_name] = len(samples)
    return counts


def convert_split(
    split_dir: Path,
    split_name: str,
    include_negatives: bool = True,
    max_negative_per_doc: int = 2,
) -> List[Dict[str, str]]:
    abstracts = read_abstracts(find_by_patterns(split_dir, ["*_abstracts.tsv", "*_abstracts_gs.tsv"]))
    entities = read_entities(find_by_patterns(split_dir, ["*_entities.tsv", "*_entities_gs.tsv"]))
    gold = read_gold_standard(find_by_patterns(split_dir, ["*_gold_standard.tsv"]))
    samples: List[Dict[str, str]] = []

    for pmid in sorted(abstracts):
        doc_entities = entities.get(pmid, {})
        chemicals = [item for item in doc_entities.values() if item.normalized_type == "chemical"]
        proteins = [item for item in doc_entities.values() if item.normalized_type == "protein"]
        negative_count = 0
        for chemical in chemicals:
            for protein in proteins:
                key = (pmid, chemical.entity_id, protein.entity_id)
                relation = gold.get(key)
                if relation:
                    samples.append(make_sample(pmid, abstracts[pmid], chemical, protein, relation, split_name))
                elif include_negatives and negative_count < max_negative_per_doc:
                    samples.append(make_sample(pmid, abstracts[pmid], chemical, protein, "NO_RELATION", split_name))
                    negative_count += 1
    return samples


def read_abstracts(path: Path) -> Dict[str, str]:
    abstracts: Dict[str, str] = {}
    for columns in iter_tsv(path):
        if len(columns) < 3:
            continue
        pmid, title, abstract = columns[0], columns[1], columns[2]
        abstracts[pmid] = f"{title} {abstract}".strip()
    return abstracts


def read_entities(path: Path) -> Dict[str, Dict[str, ChemProtEntity]]:
    entities: Dict[str, Dict[str, ChemProtEntity]] = {}
    for columns in iter_tsv(path):
        if len(columns) < 6:
            continue
        entity = ChemProtEntity(
            pmid=columns[0],
            entity_id=columns[1],
            raw_type=columns[2],
            start=columns[3],
            end=columns[4],
            mention=columns[5],
        )
        entities.setdefault(entity.pmid, {})[entity.entity_id] = entity
    return entities


def read_gold_standard(path: Path) -> Dict[Tuple[str, str, str], str]:
    relations: Dict[Tuple[str, str, str], str] = {}
    for columns in iter_tsv(path):
        if len(columns) < 4:
            continue
        pmid, label = columns[0], columns[1]
        if label not in POSITIVE_LABELS:
            continue
        arg1 = strip_arg(columns[2])
        arg2 = strip_arg(columns[3])
        relations[(pmid, arg1, arg2)] = label
        relations[(pmid, arg2, arg1)] = label
    return relations


def make_sample(
    pmid: str,
    text: str,
    chemical: ChemProtEntity,
    protein: ChemProtEntity,
    relation: str,
    split_name: str,
) -> Dict[str, str]:
    return {
        "id": f"ChemProt_{split_name}_{pmid}_{chemical.entity_id}_{protein.entity_id}",
        "text": text,
        "head_entity": chemical.mention,
        "head_type": "chemical",
        "tail_entity": protein.mention,
        "tail_type": "protein",
        "gold_relation": relation,
        "split": split_name,
    }


def find_split_dir(root: Path, raw_name: str) -> Path:
    direct = root / raw_name
    if direct.exists():
        return direct
    matches = list(root.rglob(raw_name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find ChemProt split directory: {raw_name} under {root}")


def find_by_patterns(directory: Path, patterns: List[str]) -> Path:
    for pattern in patterns:
        matches = sorted(
            path
            for path in directory.rglob(pattern)
            if path.is_file() and "__MACOSX" not in path.parts and not path.name.startswith("._")
        )
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot find any of {patterns} under {directory}")


def iter_tsv(path: Path) -> Iterable[List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            yield line.split("\t")


def strip_arg(value: str) -> str:
    return value.split(":", 1)[1] if ":" in value else value


def write_jsonl(path: Path, records: Iterable[Dict[str, str]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
