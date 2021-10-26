import torch
import torch.nn as nn
from modules.feedbackmemory import FeedbackMemory
from modules.feedbackposition import FeedbackPosition
from modules.feedbackattention import FeedbackAttention
from modules.gatedfeedforward import GatedFeedForward

class FeedbackLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            d_head: int,
            d_inner: int,
            drop_out: float,
            drop_att: float,
            shared_kv_proj: nn.Linear
    ):
        super(FeedbackLayer, self).__init__()
        
        self.attn = FeedbackAttention(
            d_model=d_model, n_head=n_head, d_head=d_head, drop_att=drop_att, shared_kv_proj=shared_kv_proj)
        
        self.ff = GatedFeedForward(d_in=d_model, d_inner=d_inner, drop=drop_out)

        self.norm = nn.ModuleList([nn.LayerNorm(d_model), nn.LayerNorm(d_model)])

    def forward(self, x, mem: FeedbackMemory, position: FeedbackPosition) -> torch.Tensor:
        x = self.norm[0](x)
        x = x + self.attn(x=x, memory=mem, pos_emb=position)
        x = self.norm[1](x)
        x = x + self.ff(x)
        return x