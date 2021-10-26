import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import einsum
from einops import rearrange
from modules.feedbackutils import exists, safe_cat

from modules.feedbackattention import FeedbackAttention
from modules.gatedfeedforward import GatedFeedForward


class FeedbackLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt, shared_kv_proj, pre_lnorm=None):
        super(FeedbackLayer, self).__init__()
        self.d_model = d_model
        self.dec_attn = FeedbackAttention(
            dim=d_model, heads=n_head, dim_head=d_head, dropout=dropatt, shared_kv_proj=shared_kv_proj)
        self.pos_ff = GatedFeedForward(dim=d_model, dropout=dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(d_model), nn.LayerNorm(d_model)])


    def forward(self, dec_inp, r, r_w_bias=None, r_r_bias=None, mems=None, dec_attn_mask=None):
        output = self.norm[0](dec_inp)
        output = output+self.dec_attn(x=output, memory=mems, pos_emb=r)
        output = self.norm[1](output)
        output = output+self.pos_ff(output)
        return output