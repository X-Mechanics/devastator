import torch
import torch.nn as nn
from collections import namedtuple
from blur.utils.config import Config

config_model_init = Config(
    init='normal',
    init_range=0.1,
    init_std=0.02,
    proj_init_std=0.01)

def init_weight(weight):
    if config_model_init.init == 'uniform':
        nn.init.uniform_(weight, -config_model_init.init_range, config_model_init.init_range)
    elif config_model_init.init == 'normal':
        nn.init.normal_(weight, 0.0, config_model_init.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('AdaptiveInput') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, 0.01)

    elif classname.find('AdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster') and m.cluster is not None:
            init_weight(m.cluster.weight)
            init_bias(m.cluster.bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, 0.01)

BlurOutput = namedtuple('BlurOutput', ['embedding', 'hiddens', 'loss'])

class BlurNew(nn.Module):
    def __init__(self, embedder, transformer, predictor, tie_weight=True):
        super(BlurNew, self).__init__()
        self.embedder = embedder
        self.transformer = transformer
        self.predictor = predictor

        if tie_weight:
            self._share_weights()

        self._init_weights()

    def forward(self, x, y, output_hidden_states=False):
        # x.shape = (b, l_q)
        # y.shape = (b, l_q)


        x = self.embedder(x)
        x = self.transformer(x)

        x_output = x.output[..., -y.size(-1):, :]
        x_output = self.predictor(x_output.view(-1, x_output.size(-1)), y.view(-1))

        output = BlurOutput(
            x.output,
            x.hiddens if output_hidden_states else None,
            x_output.loss
        )

        return output

    def reset_memories(self):
        self.transformer.reset_memories()

    def _init_weights(self):
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.embedder.apply(weights_init)
        self.predictor.apply(weights_init)

    def _share_weights(self):
        # sharing the projection layers
        for i in range(len(self.embedder.cutoffs) - 1):
            self.embedder.head.weight = self.predictor.head.weight

            for i in range(len(self.embedder.cutoffs) - 1):
                self.embedder.tail[i][0].weight = self.predictor.tail[i][0].weight
            #
            # self.embedder.tail[i][0].weight = self.predictor.tail[i][1].weight
            # self.embedder.tail[i][1].weight = torch.nn.Parameter(
            #     self.predictor.tail[i][0].weight.transpose(0, 1)
            # )
