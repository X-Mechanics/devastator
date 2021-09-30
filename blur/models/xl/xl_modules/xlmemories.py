import torch
from torch import nn
from typing import List
from copy import deepcopy

class XlMemories(nn.Module):
    def __init__(self, n_layer: int, l_q: int, l_k: int):
        assert l_k >= l_q, 'l_k !>= l_q'
        super(XlMemories, self).__init__()
        self.n_layer = n_layer
        self.l_q = l_q
        self.l_k = l_k
        self.fragments = list()

    def __getitem__(self, item: int) -> torch.Tensor:
        return self.fragments[item]

    def __iter__(self):
        return iter(self.fragments)

    def is_empty(self):
        return len(self.fragments)==0

    def init_memory(self):
        del self.fragments

        self.fragments = [torch.empty(0) for _ in range(self.n_layer + 1)]

    def update_memory(self, hiddens: List[torch.Tensor]):
        assert len(hiddens) == len(self.fragments), 'len(hiddens) != len(memories)'
        del self.fragments

        self.fragments = hiddens[:]