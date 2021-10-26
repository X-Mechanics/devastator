import torch.nn as nn
import torch.nn.functional as F


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x

class GatedFeedForward(nn.Module):
    def __init__(self, d_in: int, d_inner: int, drop=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 2 * d_inner),
            GEGLU(),
            nn.Dropout(drop),
            nn.Linear(d_inner, d_in)
        )

    def forward(self, x):
        return self.net(x)