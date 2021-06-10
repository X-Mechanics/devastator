import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, demb: int, clamp_len: int=-1):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        self.clamp_len = clamp_len

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def get_seq(self, klen, device, dtype):
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=device, dtype=dtype)

        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        return pos_seq

    def forward(self, pos_seq, bsz: int=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]