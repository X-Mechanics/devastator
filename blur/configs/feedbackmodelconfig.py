from dataclasses import dataclass

@dataclass
class FeedbackModelConfig:
    n_layer: int = 8
    d_model: int = 512
    n_head: int = 8
    d_head: int = 128
    d_inner: int = 2048
    drop_out: float = 0.1
    drop_att: float = 0.0
    tgt_len: int = 32
    mem_len: int = 256
    same_length = False