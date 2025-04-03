import math
import torch

class MultiHeadSelfAtt(nn.Module):
  def __init__(self, hidden, head):
    super().__init__()
    dim = hidden * head
    self.dim = dim
    self.LQ = nn.Linear(dim, dim)
    self.LK = nn.Linear(dim, dim)
    self.LV = nn.Linear(dim, dim)
    self.L = nn.Linear(dim, dim)
    self.norm = nn.LayerNorm(dim)
    self.small_norm = nn.LayerNorm(hidden)
    self.activ = nn.ReLU()
    self.hidden = hidden
    self.head = head
  
  def forward(self, x):
    old_shape = x.shape
    new_shape = x.shape[:-1] + (self.head, self.hidden)
    scale = math.sqrt(self.hidden)
    # like in energy former, we iterate a few times to minmize energy
    Q = self.LQ(x).reshape(new_shape)
    K = self.LK(x).reshape(new_shape)
    V = self.LV(x).reshape(new_shape)
    dot = torch.einsum('blhi,bmhi->bhlm', K, Q)
    exp = torch.softmax(dot/scale, dim=-1)
    att = torch.einsum('bhlm,bmhi->blhi', exp, V)
    att = att.reshape(old_shape)
    x = x + self.norm(att)
    x = self.norm(x)
    # linear and activation
    x = self.norm(x + self.activ(self.L(x)))
    return x

class Antiformer(nn.Module):
  def __init__(self, hidden, head, nlayer):
    super().__init__()
    self.head = head
    self.hidden = hidden
    self.linear = nn.Linear(3, hidden*head)
    self.layers = nn.ModuleList()
    for _ in range(nlayer):
      self.layers.append(MultiHeadSelfAtt(hidden, head))

    self.triu_idx = torch.triu_indices(hidden, hidden, offset=1)
    # size nhead * (hidden - 1) * hidden / 2
    self.skew = nn.Parameter(torch.randn(head, self.triu_idx.shape[-1]))

  def skew_product(self, x, y):
    A = torch.zeros(self.head, self.hidden, self.hidden).to(x.device)
    A[:, self.triu_idx[0], self.triu_idx[1]] = self.skew
    A[:, self.triu_idx[1], self.triu_idx[0]] = -self.skew
    new_shape = x.shape[:-1] + (self.head, self.hidden)
    return torch.einsum('hxy,bihx,bjhy->bhij', A, x.reshape(new_shape), y.reshape(new_shape))
  
  def forward(self, x, alpha = 0.1, niter = 1):
    x = self.linear(x)
    for layer in self.layers:
      x = layer(x)

    skew = self.skew_product(x, x)
    length = skew.shape[-1]
    idx = torch.triu_indices(length, length, offset=1)

    # yields a tensor of size (batch, length)
    # each entry is antisymmetric, so one can sum it or simply take the first entry to create a wave-function
    flat = torch.tanh(skew[:, :, idx[0], idx[1]] * self.hidden).prod(dim = -1)

    return flat.sum(dim = -1)
