import torch
import torch.nn as nn

from modules.adaptiveinput import AdaptiveInput
from modules.adaptivelogsoftmax import AdaptiveLogSoftmax
from modules.feedbackmemories import FeedbackMemory
from modules.feedbackutils import exists
from feedbacklayer import RelativePositionBias
from feedback import Feedback

from xlinitializer import XlInitializer as weights_init


class BlurFeedback(nn.Module):
    def __init__(
            self,
            n_token, n_layer, n_head, d_model, d_head, d_inner, drop_out, drop_att,
            tie_weight=True, d_embed=None, div_val=1,
            tgt_len=None, mem_len=None, ext_len=None,
            cutoffs=[], same_length=False, clamp_len=-1
    ):
        super(BlurFeedback, self).__init__()
        self.n_token = n_token
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_head = d_head

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

        self.pos_emb = RelativePositionBias(causal=True, heads=n_head)
        self.encoder = AdaptiveInput(d_model=d_model, n_classes=n_token, cutoffs=cutoffs, div_value=div_val)
        self.transformer = Feedback(
            n_layer, n_head, d_model, d_head=d_head, d_inner=d_inner, dropout=drop_out, dropatt=drop_att,
            mem_len=mem_len, seq_len=tgt_len, keep_last_hidden=False, same_length=same_length, pre_lnorm=False
        )
        self.lm_loss = AdaptiveLogSoftmax(d_model=d_model, n_classes=n_token, cutoffs=cutoffs, div_value=div_val)

        if tie_weight:
            self._share_weights()

        self._init_weights()
        self.param_dtype = next(self.parameters()).dtype


    def _init_weights(self):
        self.apply(weights_init)
        self.encoder.apply(weights_init)  # ensure embedding init not overridden by weight sharing

    def _share_weights(self):
        self.encoder.head.weight = self.lm_loss.head.weight
        for i in range(len(self.encoder.cutoffs) - 1):
            self.encoder.tail[i].weight = self.lm_loss.tail[i].weight


    def compute_loss(self, core_out, target):
        tgt_len = target.size(0)
        pred_hid = core_out[-tgt_len:]

        output = self.lm_loss(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
        return -output.output.view(tgt_len, -1)

    def forward(self, x, y, memory=None, return_memory=False):
        b, n, device = *x.shape, x.device
        x = self.encoder(x)

        outputs = []

        # calculate weighting of layers for storing to memory

        for x in x.split(self.mem_len, dim=1):
            dec_outp = self.transformer(dec_inp=x, pos_emb=self.pos_emb, mems=memory)
            outputs.append(dec_outp['output'])
            memory = dec_outp['mems']

        x = torch.cat((outputs), dim=1)

        output = self.lm_loss(x.view(-1, x.size(-1)), y.view(-1))
        loss = -output.output.view(y.size(1), -1)

        return loss, memory