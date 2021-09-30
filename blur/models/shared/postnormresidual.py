from torch import nn

class PostNormResidual(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, d: int, drop: float):
        super(PostNormResidual, self).__init__()
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, sublayer: nn.Module):
        return self.norm(x + sublayer(x))
    #     return self.norm(x + self.dropout(sublayer(x)))