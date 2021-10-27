from dataclasses import dataclass

@dataclass
class FeedbackConfig:
    n_layer: int = 6
    d_model: int = 512
    n_head: int = 10
    d_head: int = 41
    d_inner: int = 2048
    drop_out: float = 0.1
    drop_att: float = 0.0
    tgt_len: int = 400
    mem_len: int = 100
    same_length = False