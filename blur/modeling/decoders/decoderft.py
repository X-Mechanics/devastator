import torch
import torch.nn as nn

from blur.modeling.modules import PositionwiseFF, RelMultiHeadAttn

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from blur.modeling.decoders.decoderxl import DecoderXLLayer


class CausalFT(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=400, same_length=True, pre_lnorm=False):
        super(CausalFT, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.same_length = same_length
        self.pre_lnorm = pre_lnorm

    def make_ft_matrix(self, qlen, klen):
        ft_len = klen - qlen + 1
        ft_mat = torch.exp(
            2 * np.pi * 1j * torch.outer(torch.arange(0, qlen), torch.arange(0, ft_len)) / ft_len)
        ft_mat = F.pad(ft_mat, (klen - ft_len, 0))
        return ft_mat / np.sqrt(ft_len)

    def get_ft_matrix(self, qlen, klen):
        matrix = self.make_ft_matrix(qlen, klen)

        for i in range(qlen):
            matrix[i] = torch.roll(matrix[i], -(qlen - i - 1))

        matrix = torch.tril(matrix, (klen - qlen))

        if self.same_length:
            matrix = torch.triu(matrix, 0)
        return matrix

    def forward_ft(self, dec_inp, qlen, klen):
        ft_matrix = self.get_ft_matrix(qlen, klen).to(dec_inp.device)
        dec_inp = torch.einsum('ml,lbd->mbd', ft_matrix, dec_inp.type_as(ft_matrix))

        return dec_inp.real#, torch.fft.fft(dec_inp, dim=-1).real

    def forward(self, dec_inp, pos_emb, add_position, mems=None):
        qlen, _, _ = dec_inp.size()
        mlen = mems.size(0) if mems is not None else 0
        klen = mlen + qlen

        if mems is not None:
            cat = torch.cat([mems, dec_inp], 0)

        if add_position:
            cat = self.dropout(cat + pos_emb)
        else:
            cat = self.dropout(cat)
        dec_inp = self.norm(dec_inp + self.forward_ft(cat, qlen, klen) / np.sqrt(klen))
        return dec_inp

class DecoderFTLayer(nn.Module):
    def __init__(self, d_model, d_inner, dropout, same_length, pre_lnorm):
        super(DecoderFTLayer, self).__init__()

        self.ft = CausalFT(d_model, same_length=same_length, pre_lnorm=pre_lnorm)
        self.pos_ff = PositionwiseFF(
            d_model=d_model, d_inner=d_inner,
            dropout=dropout, pre_lnorm=pre_lnorm)

    def forward(self, dec_inp, pos_emb, add_position, mems=None):
        output = self.ft(dec_inp, pos_emb, add_position, mems=mems)
        output = self.pos_ff(output)

        return output

class DecoderFT(nn.Module):
    def __init__(
            self, n_layer, n_head, d_model, d_head, d_inner, dropout, dropatt,
            same_length=False, pre_lnorm=False, nft=6, nxl=6, ft_first=False):
        super(DecoderFT, self).__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.drop = nn.Dropout(dropout)
        self.same_length = same_length

        self.layers_ft = nn.ModuleList()
        self.layers_xl = nn.ModuleList()

        for _ in range(nft):
            self.layers_ft.append(
                CausalFT(d_model, same_length=True, pre_lnorm=pre_lnorm))

        for _ in range(nxl):
            self.layers_xl.append(
                DecoderXLLayer(n_head, d_model, d_head, d_inner, dropout, dropatt, pre_lnorm))

        self.n_layer = nft + nxl
        self.ft_first = ft_first

        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))


    def make_mask(self, dec_inp, qlen, mlen, klen):
        if self.same_length:
            all_ones = dec_inp.new_ones(qlen, klen)
            mask_len = klen - mlen
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).bool()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                dec_inp.new_ones(qlen, klen), diagonal=1+mlen).bool()[:,:,None]
        return dec_attn_mask

    def forward(self, dec_inp, pos_emb, mems, dec_attn_mask=None):
        qlen, _, _ = dec_inp.size()
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        if dec_attn_mask is None:
            dec_attn_mask = self.make_mask(dec_inp, qlen, mlen, klen)
            
        dec_inp, pos_emb = self.drop(dec_inp), self.drop(pos_emb)

        hidden = []
        hidden.append(dec_inp)

        ilayer = 0

        if self.ft_first:
            for i, layer in enumerate(self.layers_ft):
                mems_i = None if mems is None else mems[ilayer]
                add_position = True if i==0 else False
                dec_inp = layer(
                    dec_inp, pos_emb, add_position=add_position, mems=mems_i)
                hidden.append(dec_inp)
                ilayer += 1

            for i, layer in enumerate(self.layers_xl):
                mems_i = None if mems is None else mems[ilayer]
                dec_inp = layer(
                    dec_inp, pos_emb, self.r_w_bias, self.r_r_bias,
                    dec_attn_mask=dec_attn_mask, mems=mems_i)
                hidden.append(dec_inp)
                ilayer += 1
        else:
            for i, layer in enumerate(self.layers_xl):
                mems_i = None if mems is None else mems[ilayer]
                dec_inp = layer(
                    dec_inp, pos_emb, self.r_w_bias, self.r_r_bias,
                    dec_attn_mask=dec_attn_mask, mems=mems_i)
                hidden.append(dec_inp)
                ilayer += 1

            for i, layer in enumerate(self.layers_ft):
                mems_i = None if mems is None else mems[ilayer]
                add_position = True if i==0 else False
                dec_inp = layer(
                    dec_inp, pos_emb, add_position=add_position, mems=mems_i)
                hidden.append(dec_inp)
                ilayer += 1







        output = {
            'output': self.drop(hidden[-1]),
            'mems': mems,
            'hidden_states': hidden
        }
        return output