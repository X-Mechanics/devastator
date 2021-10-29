from dataclasses import dataclass

@dataclass
class FeedbackOptimizerConfig:
    max_step: int = 200000
    eta_min: float = 0.0
    clip: float = 0.1
    lr_min: float = 0.0
    decay_rate: float = 0.5
    warmup_step: int = 8000
    scheduler: str = 'inv_sqrt'
    lr: float = 0.0007
    optim: str = 'adam'