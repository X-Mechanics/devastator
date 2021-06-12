import torch.nn as nn
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
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)

    elif classname.find('AdaptiveInput') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, config_model_init.proj_init_std)

    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)

    elif classname.find('AdaptiveLogSoftmaxWithLoss') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, config_model_init.proj_init_std)

    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, config_model_init.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)

    elif classname.find('QNet') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = config_model_init.dropout

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = config_model_init.dropatt