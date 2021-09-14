import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

class XlPosition(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_head: int, clamp_len: int=-1):
        super(XlPosition, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.clamp_len = clamp_len

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))


        self.register_buffer('inv_freq', self._inv_freq())

    def forward(self, w_head_q, w_head_k, r, rlen):
        r_head_k = self.r_net(r)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        C = einsum('nd,jbnd->jn', self.r_w_bias, w_head_k)  # qlen x klen x bsz x n_head
        C = rearrange(C, 'j n -> () j () n')
        B = einsum('ibnd,jnd->ijbn', w_head_q, r_head_k)  # qlen x klen x bsz x n_head
        D = einsum('nd,jnd->jn', self.r_r_bias, r_head_k)  # qlen x klen x bsz x n_head
        D = rearrange(D, 'j n -> () j () n')
        BD = self._rel_shift(B+D)

        # [qlen x klen x bsz x n_head]
        return C + BD

    def embed(self, pos_seq, bsz: int=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq.to(pos_seq.device))
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]

    def get_seq(self, klen, device, dtype):
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=device, dtype=dtype)

        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        return pos_seq

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def _inv_freq(self):
        inv_freq = 1 / (10000 ** (torch.arange(0.0, self.d_model, 2.0) / self.d_model))
        return inv_freq

class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        self.pos_emb = XlPosition(d_model=d_model, d_head=d_head, n_head=n_head)

    def forward(self, w, r, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)        # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)        # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)        # qlen x bsz x n_head x d_head

        # r_head_k = self.r_net(r)
        # r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)             # qlen x n_head x d_head

        #### compute attention score
        # rw_head_q = w_head_q + r_w_bias                                      # qlen x bsz x n_head x d_head
        # AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))          # qlen x klen x bsz x n_head
        #
        # rr_head_q = w_head_q + r_r_bias
        # BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))           # qlen x klen x bsz x n_head
        # BD = self._rel_shift(BD)
        #
        # # [qlen x klen x bsz x n_head]
        attn_score = einsum('i b n d, j b n d -> i j b n', w_head_q, w_head_k)
        attn_score = attn_score + self.pos_emb(w_head_q, w_head_k, r, rlen)
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

