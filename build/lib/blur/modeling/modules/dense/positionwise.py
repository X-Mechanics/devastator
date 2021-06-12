import torch
import torch.nn as nn


class PositionwiseFF(nn.Module):
    def __init__(self, d_model: int, d_inner: int, dropout: float, pre_lnorm: bool=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp: torch.tensor):
        if self.pre_lnorm:
            core_out = self.ff(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.ff(inp)
            output = self.layer_norm(inp + core_out)

        return output