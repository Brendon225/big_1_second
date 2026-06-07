from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ModelOutput:
    loss: float
    generation_loss: float
    alignment_loss: float
    predictions: List[Dict[str, Any]]
    prototype_scores: List[Dict[str, Any]] = field(default_factory=list)
