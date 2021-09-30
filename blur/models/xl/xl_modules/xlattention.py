import torch
from torch import nn
import torch.nn.functional as F
from blur.models.shared import Attention

class XlAttention(Attention):
    def __init__(self, d_in: int, d_head: int, n_head: int, drop_out: float, drop_att: float):
        super(XlAttention, self).__init__(
            d_in_q=d_in,
            d_in_k=d_in,
            d_out=d_in,
            d_head=d_head,
            n_head=n_head,
            drop_out=drop_out,
            drop_attn=drop_att
        )
        self.R = nn.Parameter(torch.Tensor(d_in, n_head, d_head))
        self.same_length = False
        self.scale = 1 / (d_head ** 0.5)

    def forward(self, x, q_len: int, position: nn.Module, mask: nn.Module):
        # x.shape = (b, l, d)
        k_len = x.size(-2)

        q = self.forward_Q(x[..., -q_len:, :])
        k, v = self.forward_KV(x)
        r = self.forward_R(position.embedding(k_len=k_len, dtype=x.dtype, device=x.device))

        attn_score = self.outer(q, k)
        # attn_score = self.scale * (attn_score )
        attn_score = self.scale * (attn_score + position(q, k, r))
        attn_score = mask(attn_score)

        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.drop_attn(attn_prob)

        x = self.score(attn_prob, v)
        x = self.output(self.O, x)
        x = self.drop_out(x)
        return x

    def forward_R(self, x):
        return self.embed(self.R, x)