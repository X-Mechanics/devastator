import torch
from torch import nn, einsum
from typing import Union


class XlPosition(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_head: int, clamp_len: int = -1):
        super(XlPosition, self).__init__()
        self.d_model = d_model
        self.clamp_len = clamp_len

        self.r_k = nn.Parameter(torch.Tensor(n_head, d_head))
        self.r_r = nn.Parameter(torch.Tensor(n_head, d_head))

        inverse = 1 / (10000 ** (torch.arange(0.0, self.d_model, 2.0) / self.d_model))
        self.register_buffer('inverse', inverse)

    def forward(self, q: torch.Tensor, k: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # q.shape = (b, l_q, n_head, d_head)
        # k.shape = (b, l_k, n_head, d_head)
        # r.shape = (b, l_k, n_head, d_head)

        C = einsum('ha, ...mha -> ...hm', self.r_k, k)[..., :, None, :]
        B = einsum('...lha, ...mha -> ...hlm', q, r)
        D = einsum('ha, ...mha -> ...hm', self.r_r, r)[..., :, None, :]
        CBD = C + self._rel_shift(B + D)

        return CBD

    def embedding(self, k_len: int, dtype: torch.dtype, device: Union[str, int]):
        sequence = torch.arange(k_len - 1, -1, -1.0, device=device, dtype=dtype)

        if self.clamp_len > 0:
            sequence.clamp_(max=self.clamp_len, min=0)

        sinusoid_inp = torch.outer(sequence, self.inverse.type_as(sequence).to(device))
        embedding = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        return embedding[None, :, :]

    def _rel_shift(self, x, zero_triu=False):
        # x.shape (b, n_head, l_q, l_k)

        zero_pad = torch.zeros((*x.size()[: -2], x.size(-2), 1), device=x.device, dtype=x.dtype)

        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(*x.size()[: -2], x.size(-1) + 1, x.size(-2))
        x = x_padded[..., 1:, :].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(-2), x.size(-1)))
            x = x * torch.tril(ones, x.size(-1) - x.size(-2))[None, None, :, :]

        return x