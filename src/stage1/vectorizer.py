import hashlib
import math
import re
from typing import Dict, Iterable, List


TOKEN_RE = re.compile(r"[a-z0-9:]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def vectorize_text(text: str, dim: int = 64) -> List[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dim
        vector[index] += 1.0
    return l2_normalize(vector)


def l2_normalize(vector: Iterable[float]) -> List[float]:
    values = list(vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def add_vectors(left: Iterable[float], right: Iterable[float]) -> List[float]:
    return [a + b for a, b in zip(left, right)]


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def softmax_cross_entropy(scores: Dict[str, float], gold_label: str) -> float:
    if gold_label not in scores:
        raise ValueError(f"Gold label not found in scores: {gold_label}")
    max_score = max(scores.values())
    total = sum(math.exp(score - max_score) for score in scores.values())
    log_prob = scores[gold_label] - max_score - math.log(total)
    return -log_prob


def stable_label_vector(label: str, dim: int = 64, scale: float = 0.05) -> List[float]:
    base = vectorize_text(label, dim=dim)
    return [value * scale for value in base]
