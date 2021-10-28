from dataclasses import dataclass

@dataclass
class RunConfig:
    work_dir: float = 'LM-TFM'
    cuda: bool = True
    seed: int = 1111
    log_interval: int = 200
    eval_interval: int = 1000
    debug: bool = False
    max_eval_steps: int = -1