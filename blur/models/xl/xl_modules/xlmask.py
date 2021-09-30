import torch
from torch import nn


class XlMask(nn.Module):
    """
    Creates, registers, and applies the attention score mask
    """

    def __init__(self, l_q: int, l_k: int, same_length: bool):
        super(XlMask, self).__init__()
        self.l_q = l_q
        self.l_k = l_k
        self.same_length = same_length

        self.attn_mask = self.make_mask()
        # self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        # x.shape = (1, 1, l_q, l_k)
        mask = self.attn_mask[None, None, :, -x.size(-1):]
        # print(x.shape, mask.shape)
        x.masked_fill_(mask.to(x.device),-torch.finfo(x.dtype).max)
        return x

    def make_mask(self):
        causal_mask = torch.ones(self.l_q, self.l_k).triu_(self.l_k - self.l_q + 1).bool()

        if self.same_length:
            causal_mask = causal_mask + torch.ones(self.l_q, self.l_k).tril_(0).bool()
        return causal_mask