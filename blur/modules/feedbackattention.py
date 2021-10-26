import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from modules.feedbackutils import exists, safe_cat


class FeedbackAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            d_head: int,
            drop_att: float,
            shared_kv_proj: nn.Linear
    ):
        super().__init__()
        self.dim = d_model
        self.heads = n_head
        self.scale = d_head ** -0.5

        inner_dim = d_head * n_head
        self.to_q = nn.Linear(d_model, inner_dim, bias=False)
        self.to_kv = shared_kv_proj
        self.to_out = nn.Linear(inner_dim, d_model)

        self.drop_att = nn.Dropout(drop_att)

    def forward(self, x, memory, pos_emb=None):
        h, n, device = self.heads, x.shape[1], x.device

        q = self.to_q(x) * self.scale

        k, v = memory if exists(memory) else (None, None)

        self_k, self_v = self.to_kv(x).chunk(2, dim=-1)
        k = safe_cat(k, self_k, dim=1)
        v = safe_cat(v, self_v, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        i, j = sim.shape[-2:]

        if exists(pos_emb):
            sim = sim + pos_emb(sim)

        causal_mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()
        causal_mask = rearrange(causal_mask, 'i j -> () () i j')
        mask_value = -torch.finfo(q.dtype).max
        sim.masked_fill_(causal_mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.drop_att(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)