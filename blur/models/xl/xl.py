

import torch
import torch.nn as nn

from blur.models.shared import FeedForward, PostNormResidual, PreNormResidual
from blur.models.xl.xl_modules import XlAttention, XlPosition, XlMask, XlMemories


class XlLayer(nn.Module):
    def __init__(self, d_model, d_head, n_head, d_hidden, drop_out, drop_att):
        super(XlLayer, self).__init__()
        self.attn = XlAttention(
            d_in=d_model, d_head=d_head, n_head=n_head, drop_out=drop_out, drop_att=drop_att)
        self.ff = FeedForward(d_in=d_model, d_hidden=d_hidden, drop=drop_out)

        self.sublayer = nn.ModuleList([
            PostNormResidual(d=d_model, drop=drop_out),
            PostNormResidual(d=d_model, drop=drop_out)
        ])

    def forward(self, x, memory: torch.Tensor, position: nn.Module, mask: nn.Module):
        # x.shape = (b, l_q, d_model)

        q_len = x.size(-2)
        # x_mem = x
        x_mem = torch.cat([memory.detach().to(x.device), x], dim=-2)

        x = self.sublayer[0](x, lambda z: self.attn(x_mem, q_len=q_len, position=position, mask=mask))
        x = self.sublayer[1](x, self.ff)
        return x

from collections import namedtuple



XlOutput = namedtuple('XlOutput', ['output', 'hiddens'])

class Xl(nn.Module):
    def __init__(
            self,
            n_layer: int,
            l_q: int,
            l_k: int,
            d_model: int,
            d_head: int,
            n_head: int,
            d_hidden: int,
            drop_out: float,
            drop_att: float,
            same_length: bool = False
    ):
        super(Xl, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(n_layer):
            self.layers.append(
                XlLayer(d_model, d_head, n_head, d_hidden, drop_out, drop_att))

        self.memories = XlMemories(n_layer=n_layer, l_q=l_q, l_k=l_k)
        self.mask = XlMask(l_q=l_q, l_k=l_k, same_length=same_length)
        self.position = XlPosition(d_model=d_model, d_head=d_head, n_head=n_head)

    def forward(self, x):
        # x.shape = (b, l_q, d_model)

        if self.memories.is_empty():
            self.memories.init_memory()

        hiddens = [x]

        for memory, layer in zip(self.memories, self.layers):
            x = layer(x, memory=memory, position=self.position, mask=self.mask)
            hiddens.append(x.detach())


        self.memories.update_memory(hiddens)
        return XlOutput(x, hiddens)

    def reset_memories(self):
        self.memories.init_memory()


















# import torch
# import torch.nn.functional as F
# from torch import nn, einsum
# from typing import Union
#
#
# class XlPosition(nn.Module):
#     def __init__(self, d_model: int, d_head: int, n_head: int, clamp_len: int = -1):
#         super(XlPosition, self).__init__()
#         self.d_model = d_model
#         self.clamp_len = clamp_len
#
#         self.r_k = nn.Parameter(torch.Tensor(n_head, d_head))
#         self.r_r = nn.Parameter(torch.Tensor(n_head, d_head))
#
#         inverse = 1 / (10000 ** (torch.arange(0.0, self.d_model, 2.0) / self.d_model))
#         self.register_buffer('inverse', inverse)
#
#     def forward(self, q, k, r):
#         C = einsum('ha, ...mha -> ...hm', self.r_k, k)[..., :, None, :]
#         B = einsum('...lha, ...mha -> ...hlm', q, r)
#         D = einsum('ha, ...mha -> ...hm', self.r_r, r)[..., :, None, :]
#         CBD = C + self._rel_shift(B + D)
#
#         return CBD
#
#     def embedding(self, seq_len: int, dtype: torch.dtype, device: Union[str, int]):
#         sequence = torch.arange(seq_len - 1, -1, -1.0, device=device, dtype=dtype)
#
#         if self.clamp_len > 0:
#             sequence.clamp_(max=self.clamp_len, min=0)
#
#         sinusoid_inp = torch.outer(sequence, self.inverse.type_as(sequence).to(device))
#         embedding = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
#
#         return embedding[None, :, :]
#
#     def _rel_shift(self, x, zero_triu=False):
#         zero_pad = torch.zeros((*x.size()[: -2], x.size(-2), 1), device=x.device, dtype=x.dtype)
#
#         x_padded = torch.cat([zero_pad, x], dim=-1)
#         x_padded = x_padded.view(*x.size()[: -2], x.size(-1) + 1, x.size(-2))
#         x = x_padded[..., 1:, :].view_as(x)
#
#         if zero_triu:
#             ones = torch.ones((x.size(-2), x.size(-1)))
#             x = x * torch.tril(ones, x.size(-1) - x.size(-2))[None, None, :, :]
#
#         return x
#
#
# # from blur.models.shared import Attention
#
# class XlAttention(Attention):
#     def __init__(self, d_in: int, d_head: int, n_head: int, drop_out: float, drop_att: float):
#         super(XlAttention, self).__init__(
#             d_in_q=d_in,
#             d_in_k=d_in,
#             d_out=d_in,
#             d_head=d_head,
#             n_head=n_head,
#             drop_out=drop_out,
#             drop_attn=drop_att
#         )
#         self.R = nn.Parameter(torch.Tensor(d_in, n_head, d_head))
#         self.same_length = False
#         self.scale = 1 / (d_head ** 0.5)
#
#     def forward(self, x, q_len: int, position: nn.Module, mask: nn.Module):
#         # x.shape = (b, l, d)
#         k_len = x.size(-2)
#
#         q = self.forward_Q(x[..., -q_len:, :])
#         k, v = self.forward_KV(x)
#         r = self.forward_R(position.embedding(seq_len=k_len, dtype=x.dtype, device=x.device))
#
#         attn_score = self.outer(q, k)
#         attn_score = self.scale * (attn_score + position(q, k, r))
#         attn_score = mask(attn_score)
#
#         attn_prob = F.softmax(attn_score, dim=1)
#         attn_prob = self.drop_attn(attn_prob)
#
#         x = self.score(attn_prob, v)
#         x = self.output(self.O, x)
#         x = self.drop_out(x)
#         return x
#
#     def forward_R(self, x):
#         return self.embed(self.R, x)
#
#
# from torch import nn
#
#
# class XlMask(nn.Module):
#     """
#     Creates, registers, and applies the attention score mask
#     """
#
#     def __init__(self, l_q: int, l_k: int, same_length: bool):
#         assert l_k >= l_q, 'l_k must be greater than l_q'
#         super(XlMask, self).__init__()
#         self.l_q = l_q
#         self.l_k = l_k
#         self.same_length = same_length
#
#         attn_mask = self.make_mask()
#         self.register_buffer('attn_mask', attn_mask)
#
#     def forward(self, x):
#         x.masked_fill_(self.attn_mask.to(x.device), -torch.finfo(x.dtype).max)
#         return x
#
#     def make_mask(self):
#         causal_mask = torch.ones(self.l_q, self.l_k).triu_(self.l_k - self.l_q + 1).bool()
#
#         if self.same_length:
#             causal_mask = causal_mask + torch.ones(self.l_q, self.l_k).tril_(0).bool()
#         return causal_mask
#
# from typing import List
# from copy import deepcopy
#
# class XlMemories(nn.Module):
#     def __init__(self, n_layer: int, l_q: int, l_k: int):
#         assert l_k >= l_q, 'l_k !>= l_q'
#         super(XlMemories, self).__init__()
#         self.n_layer = n_layer
#         self.l_q = l_q
#         self.l_k = l_k
#         self.fragments = list()
#
#     def __getitem__(self, item: int) -> torch.Tensor:
#         return self.fragments[item]
#
#     def __iter__(self):
#         return iter(self.fragments)
#
#     def is_empty(self):
#         return len(self.fragments) == 0
#
#     def update_memory(self, hiddens: List[torch.Tensor]):
#         assert len(hiddens) == len(self.fragments), 'len(hids) != len(mems)'
#         del self.fragments
#         self.fragments = deepcopy(hiddens)
#
#
# import torch
# import torch.nn as nn
#
# from blur.models.shared import FeedForward, PostNormResidual, PreNormResidual
#
#
# class XlLayer(nn.Module):
#     def __init__(self, d_model, d_head, n_head, d_hidden, dropout, dropatt):
#         super(XlLayer, self).__init__()
#         self.attn = XlAttention(d_in=d_model, d_head=d_head, n_head=n_head, drop_out=dropout, drop_att=dropatt)
#         self.ff = FeedForward(d_in=d_model, d_hidden=d_hidden, drop=dropout)
#
#         self.sublayer = nn.ModuleList([
#             PreNormResidual(d=d_model, drop=dropout),
#             PreNormResidual(d=d_model, drop=dropout)
#         ])
#
#     def forward(self, x, memory: torch.Tensor, position: nn.Module):
#         q_len = x.size(-2)
#         x_mem = torch.cat([memory, x], dim=-2)
#
#         x = self.sublayer[0](x, lambda z: self.attn(x_mem, q_len, position))
#         x = self.sublayer[1](x, self.ff)
#         return x
#
# from collections import namedtuple
# XlOutput = namedtuple('XlOutput', ['output', 'memory', 'hiddens'])
#
#
# class Xl(nn.Module):
#     def __init__(
#             self,
#             n_layer: int,
#             l_q: int,
#             l_k: int,
#             d_model: int,
#             d_head: int,
#             n_head: int,
#             d_hidden: int,
#             dropout: float,
#             dropatt: float,
#             same_length: bool = False
#     ):
#         super(Xl, self).__init__()
#         self.layers = nn.ModuleList()
#
#         for i in range(n_layer):
#             self.layers.append(
#                 XlLayer(d_model, d_head, n_head, d_hidden, dropout, dropatt))
#
#         self.position = XlPosition(d_model=d_model, d_head=d_head, n_head=n_head)
#         self.mask = XlMask(l_q=l_q, l_k=l_k)
#         self.memories = XlMemories()
#
#         self.drop = nn.Dropout(dropout)
#         self.same_length = same_length
#
#     def forward(self, x, pos_emb, mems):
#         x, pos_emb = self.drop(x), self.drop(pos_emb)
#
#         hidden = []
#         hidden.append(x)
#
#         for memory, layer in zip(self.memories, self.layers):
#             x = layer(x, memory=memory, position=self.position)
#             hidden.append(x.detach())
#
#         return XlOutput(x, mems, hidden)
#
#
# class BlurXl(nn.Module):
#     def __init__(self, config_model, config_encoder, config_decoder, tie_weight=True, clamp_len=-1):
#         super(BlurXl, self).__init__()
#         self.tgt_len = config_model
#         self.mem_len = mem_len
#         self.ext_len = ext_len
#
#         self.embedder = AdaptiveInput(**config_encoder.parameters())
#         self.transformer = Xl(**config_decoder.parameters())
#         self.predictor = AdaptiveLogSoftmaxWithLoss(**config_encoder.parameters())
#
#         if tie_weight:
#             self._share_weights()
#
#         self._init_weights()
#         self.param_dtype = next(self.parameters()).dtype
#
#     def forward(self, data, target, mems, dec_attn_mask=None, output_hidden_states=False):
#
#         if mems is None or len(mems)= =0:
#             # if not mems:
#             mems = self._init_mems(device=data.device)
#         qlen, _ = data.size()
#         mlen = mems[0].size(0) if mems is not None else 0
#         klen = mlen + qlen
#
#         pos_seq = self.pos_emb.get_seq(klen=klen, device=data.device, dtype=data.dtype)
#         pos_emb = self.pos_emb(pos_seq)
#         dec_inp = self.embedder(data)
#
#         dec_outp = self.transformer(dec_inp, pos_emb, mems=mems, dec_attn_mask=dec_attn_mask)
#         output = {
#             'output': dec_outp['output'],
#             'mems': self._update_mems(dec_outp['hidden_states'], mems, qlen, mlen),
#             'loss': self.compute_loss(dec_outp['output'], target)
#         }
#
#         if output_hidden_states:
#             output['hidden_states'] = dec_outp['hidden_states']
#
#         return output
#
#     def compute_loss(self, core_out, target):
#         tgt_len = target.size(0)
#         pred_hid = core_out[-tgt_len:]
#
#         output = self.predictor(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
#         return -output.output.view(tgt_len, -1)
#
#     def _init_weights(self):
#         self.apply(weights_init)
#         self.embedder.apply(weights_init)  # ensure embedding init not overridden by weight sharing
#
#     def _share_weights(self):
#         # sharing the projection layers
#         for i in range(len(self.embedder.cutoffs) - 1):
#             self.embedder.tail[i][0].weight = self.predictor.tail[i][1].weight
#             self.embedder.tail[i][1].weight = torch.nn.Parameter(
#                 self.predictor.tail[i][0].weight.transpose(0, 1)
#             )
