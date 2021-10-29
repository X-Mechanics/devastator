from dataclasses import dataclass

@dataclass
class FeedbackDataConfig:
    data: str = '../data/wikitext-103'
    dataset: str = 'wt103'
    tgt_len: int = 150
    mem_len: int = None
    batch_size: int = 60
    batch_chunk: int = 15
    eval_tgt_len: int = 256
    eval_mem_len: int = 256
    eval_batch_size: int = 10
    n_layer: int = 8