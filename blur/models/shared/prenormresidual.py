from torch import nn

class PreNormResidual(nn.Module):
    """
    A layer norm followed by a residual connection.
    """

    def __init__(self, d: int, drop: float):
        super(PreNormResidual, self).__init__()
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, sublayer: nn.Module):
        return x + self.dropout(sublayer(self.norm(x)))