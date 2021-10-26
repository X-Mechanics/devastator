from typing import List

class XlAdaptiveConfig:
    d_model: int = 410
    n_classes: int = None
    cutoffs: List[int] = [20000, 40000, 200000]
    div_value: float = 1.0