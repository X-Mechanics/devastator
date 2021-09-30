import torch
from torch import nn, einsum

class Attention(nn.Module):
    def __init__(
            self,
            d_in_q: int,
            d_in_k: int,
            d_out: int,
            d_head: int,
            n_head: int,
            drop_out: float,
            drop_attn: float,
    ):
        super(Attention, self).__init__()
        self.Q = nn.Parameter(torch.Tensor(d_in_q, n_head, d_head))
        self.K = nn.Parameter(torch.Tensor(d_in_k, n_head, d_head))
        self.V = nn.Parameter(torch.Tensor(d_in_k, n_head, d_head))
        self.O = nn.Parameter(torch.Tensor(d_out, n_head, d_head))

        self.drop_attn = nn.Dropout(p=drop_attn)
        self.drop_out = nn.Dropout(p=drop_out)

    def forward_Q(self, x):
        x = self.embed(self.Q, x)
        return x

    def forward_KV(self, x):
        k = self.embed(self.K, x)
        v = self.embed(self.V, x)
        return k, v

    def embed(self, op: torch.Tensor, x) -> torch.Tensor:
        return einsum('dha, ...ld -> ...lha', op, x)

    def outer(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return einsum('...lha, ...mha -> ...hlm', q, k)

    def score(self, op: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return einsum('...hlm, ...mha -> ...lha', op, x)

    def output(self, op: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return einsum('...dha, ...lha -> ...ld', op, x)
