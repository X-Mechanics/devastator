from dataclasses import dataclass

@dataclass
class OptimizerConfig:
    max_step: int = 200000
    eta_min: float = 0.0
    clip: float = 0.25
    lr_min: float = 0.0
    decay_rate: float = 0.5
    warmup_step: int = 0
    scheduler: str = 'cosine'
    lr: float = 0.00025
    optim: str = 'adam'