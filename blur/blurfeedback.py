import torch
import torch.nn as nn

from modules.adaptiveinput import AdaptiveInput
from modules.adaptivelogsoftmax import AdaptiveLogSoftmax
from modules.feedbackmemories import FeedbackMemory
from modules.feedbackutils import exists
from models.feedback import Feedback


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

        self.encoder = AdaptiveInput(d_model=d_model, n_classes=n_token, cutoffs=cutoffs, div_value=div_val)
        self.transformer = Feedback(
            n_layer=n_layer, d_model=d_model, n_head=n_head, d_head=d_head, d_inner=d_inner,
            dropout=drop_out, dropatt=drop_att, seq_len=tgt_len, mem_len=mem_len,
            same_length=same_length
        )
        self.lm_loss = AdaptiveLogSoftmax(d_model=d_model, n_classes=n_token, cutoffs=cutoffs, div_value=div_val)

        if tie_weight:
            self._share_weights()

    def _share_weights(self):
        self.encoder.head.weight = self.lm_loss.head.weight
        for i in range(len(self.encoder.cutoffs) - 1):
            self.encoder.tail[i].weight = self.lm_loss.tail[i].weight

    def forward(self, x, y, memory=None, return_memory=False):
        x = self.encoder(x)

        x, new_memory = self.transformer(x=x, memory=memory)

        output = self.lm_loss(x.view(-1, x.size(-1)), y.view(-1))
        loss = -output.output.view(y.size(1), -1)

        return loss, new_memory