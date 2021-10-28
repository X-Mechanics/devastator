from dataclasses import dataclass

@dataclass
class XlModelConfig:
    n_layer: int = 16
    d_model: int = 410
    n_head: int = 10
    d_head: int = 41
    d_inner: int = 2100
    drop_out: float = 0.1
    drop_att: float = 0.0
    tgt_len: int = 150
    mem_len: int = 150
    same_length: bool = False
    clamp_len: int = -1