import torch
import torch.nn as nn
from blur.modeling.modules import PositionalEmbedding
from blur.modeling.initializers import weights_init

class Blur(nn.Module):
    def __init__(self, tgt_len, mem_len, ext_len, encoder, decoder, lm_loss, tie_weight=True, clamp_len=-1):
        super(Blur, self).__init__()
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

        self.pos_emb = PositionalEmbedding(demb=encoder.d_model, clamp_len=clamp_len)
        self.encoder = encoder
        self.decoder = decoder
        self.lm_loss = lm_loss

        if tie_weight:
            self._share_weights()

        self._init_weights()
        self.param_dtype = next(self.parameters()).dtype

    def _batch_first(self, t):
        return torch.einsum('i...k->k...i', t)

    def _init_weights(self):
        self.apply(weights_init)
        self.encoder.apply(weights_init)  # ensure embedding init not overridden by weight sharing

    def _share_weights(self):
        for i in range(len(self.encoder.cutoffs) - 1):
            self.encoder.tail[i][0].weight = self.lm_loss.tail[i][1].weight
            self.encoder.tail[i][1].weight = torch.nn.Parameter(
                self.lm_loss.tail[i][0].weight.transpose(0, 1)
            )  # sharing the projection layers

    def _reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def _init_mems(self, device):
        if self.mem_len > 0:
            mems = []

            for i in range(self.decoder.n_layer + 1):
                empty = torch.empty(0, dtype=self.param_dtype, device=device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def compute_loss(self, core_out, target):
        tgt_len = target.size(0)
        pred_hid = core_out[-tgt_len:]

        output = self.lm_loss(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
        return -output.output.view(tgt_len, -1)

    def forward(self, data, target, mems, dec_attn_mask=None, output_hidden_states=False):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems:
            mems = self._init_mems(device=data.device)

        data = self._batch_first(data).contiguous()
        target = self._batch_first(target).contiguous()

        qlen, _ = data.size()
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        pos_seq = self.pos_emb.get_seq(klen=klen, device=data.device, dtype=data.dtype)
        pos_emb = self.pos_emb(pos_seq)
        dec_inp = self.encoder(data)

        dec_outp = self.decoder(dec_inp, pos_emb, mems=mems, dec_attn_mask=dec_attn_mask)

        output = {
            'output': dec_outp['output'],
            'mems': self._update_mems(dec_outp['hidden_states'], mems, qlen, mlen),
            'loss': self.compute_loss(dec_outp['output'], target)
        }

        if output_hidden_states:
            output['hidden_states'] = dec_outp['hidden_states']

        return output

