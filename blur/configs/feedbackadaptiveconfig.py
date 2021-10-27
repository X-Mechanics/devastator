from dataclasses import dataclass, field
from typing import List

@dataclass
class FeedbackAdaptiveConfig:
    d_model: int = 512
    n_classes: int = None
    cutoffs: List[int] = field(default_factory=lambda: [20000, 40000, 200000])
    div_value: float = 1.0
