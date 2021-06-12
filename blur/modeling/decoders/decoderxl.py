import torch
import torch.nn as nn

from blur.modeling.modules import PositionwiseFF, RelMultiHeadAttn

class DecoderXLLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt, pre_lnorm):
        super(DecoderXLLayer, self).__init__()
        self.dec_attn = RelMultiHeadAttn(
            n_head=n_head, d_model=d_model, d_head=d_head,
            dropout=dropout, dropatt=dropatt, pre_lnorm=pre_lnorm)
        self.pos_ff = PositionwiseFF(
            d_model=d_model, d_inner=d_inner,
            dropout=dropout, pre_lnorm=pre_lnorm)

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)

        return output

class DecoderXL(nn.Module):
    def __init__(
            self, n_layer, n_head, d_model, d_head, d_inner, dropout, dropatt,
            same_length=False, pre_lnorm=False):
        super(DecoderXL, self).__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.drop = nn.Dropout(dropout)
        self.same_length = same_length

        self.layers = nn.ModuleList()

        for i in range(n_layer):
            self.layers.append(
                DecoderXLLayer(n_head, d_model, d_head, d_inner, dropout, dropatt, pre_lnorm))

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

        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            dec_inp = layer(
                dec_inp, pos_emb, self.r_w_bias, self.r_r_bias,
                dec_attn_mask=dec_attn_mask, mems=mems_i)
            hidden.append(dec_inp)

        output = {
            'output': self.drop(hidden[-1]),
            'mems': mems,
            'hidden_states': hidden
        }
        return output