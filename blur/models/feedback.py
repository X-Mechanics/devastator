import torch
import torch.nn as nn
from models.feedbacklayer import FeedbackLayer
from modules.feedbackmemories import FeedbackMemory
from modules.feedbackposition import FeedbackPosition
from modules.feedbackutils import exists, safe_cat


class Feedback(nn.Module):
    def __init__(
            self,
            n_layer: int,
            d_model: int,
            n_head: int,
            d_head: int,
            d_inner: int,
            dropout: float,
            dropatt: float,
            seq_len: int,
            mem_len: int,
            same_length=False,
    ):
        super(Feedback, self).__init__()
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.same_length = same_length

        self.position = FeedbackPosition(n_head=n_head, causal=True)

        self.shared_kv_proj = nn.Linear(d_model, 2 * d_head * n_head, bias=False)

        self.layers = nn.ModuleList([])

        for _ in range(n_layer):
            self.layers.append(FeedbackLayer(
                d_model=d_model, n_head=n_head, d_head=d_head, d_inner=d_inner,
                drop_out=dropout, drop_att=dropatt, shared_kv_proj=self.shared_kv_proj
            ))

        # memory parameters

        self.layer_weight = nn.Parameter(torch.ones(n_layer + 1))

    def get_layer_weight(self):
        layer_weight = self.layer_weight.softmax(dim=-1)
        return layer_weight[:, None, None, None]

    def forward(self, x, memory):
        outputs = []

        # calculate weighting of layers for storing to memory

        for split in x.split(self.mem_len, dim=1):
            dec_outp = self._forward(dec_inp=split, pos_emb=self.position, mems=memory)

            outputs.append(dec_outp['output'])
            memory = dec_outp['mems']

        x = torch.cat((outputs), dim=1)

        return x, memory

    def _forward(self, dec_inp, pos_emb, mems, dec_attn_mask=None):
        if exists(mems):
            memory_keys, memory_values = mems

        dec_outp = dec_inp
        # prepare memory for attention, if it exists
        hiddens = [dec_outp]

        memory = None
        if exists(memory_keys):
            memory = (memory_keys, memory_values)

        for layer in self.layers:
            dec_outp = layer(x=dec_outp, mem=memory, position=pos_emb)
            hiddens.append(dec_outp)

        # calculate new memory key / values and store to FIFO queue

        hiddens = torch.stack(hiddens)
        layer_weight = self.get_layer_weight()
        agg_hiddens = (hiddens * layer_weight).sum(dim=0)

        # pre-calculate memory key / values and store to buffer

        mem_k, mem_v = self.shared_kv_proj(agg_hiddens).chunk(2, dim=-1)
        memory_keys = safe_cat(memory_keys, mem_k, dim=1)
        memory_values = safe_cat(memory_values, mem_v, dim=1)

        # enforce max length on memory buffer

        memory_keys = memory_keys[:, -self.mem_len:]
        memory_values = memory_values[:, -self.mem_len:]

        output = {
            'output': dec_outp,
            'mems': FeedbackMemory(memory_keys, memory_values),
            'agg_hiddens': agg_hiddens
        }
        return output