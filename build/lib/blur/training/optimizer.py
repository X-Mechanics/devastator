import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from blur.utils.config import Config

class ScheduledOptimizer:
    optims = ['sgd', 'adam', 'adagrad']
    scheds = ['inv_sqrt', 'cosine']

    def __init__(self, config: Config, model):
        self.config = config
        self.optim = self._init_optimizer(model.parameters())
        self.sched = self._init_scheduler()

    def _init_optimizer(self, model_parameters):
        if self.config.optim == 'sgd':
            return optim.SGD(model_parameters, lr=self.config.lr, momentum=self.config.mom)

        elif self.config.optim == 'adam':
            return optim.Adam(model_parameters, lr=self.config.lr)

        elif self.config.optim == 'adagrad':
            return optim.Adagrad(model_parameters, lr=self.config.lr)

        else:
            raise KeyError('optimizer must be one of: {}'.format(ScheduledOptimizer.optims))

    def _init_scheduler(self):
        if self.config.scheduler == 'cosine':
            return CosineAnnealingLR(self.optim, self.config.max_step, eta_min=self.config.eta_min)

        elif self.config.scheduler == 'inv_sqrt':

            def lr_lambda(step):  # return a multiplier instead of a learning rate
                if step == 0 and self.config.warmup_step == 0:
                    return 1.0
                elif step > self.config.warmup_step:
                    return 1. / (step ** 0.5)
                else:
                    return step / (self.config.warmup_step ** 1.5)

            return LambdaLR(self.optim, lr_lambda=lr_lambda)

        else:
            return None

    def backward(self, loss):
        self.optim.backward(loss)

    def clip_master_grads(self, clip):
        self.optim.clip_master_grads(clip)

    def step(self):
        self.optim.step()

    def update(self, step):
        if self.config.scheduler == 'cosine' and step < self.config.warmup_step:
            curr_lr = self.config.lr * step / self.config.warmup_step
            self.optim.param_groups[0]['lr'] = curr_lr

        elif self.sched is not None:
            self.sched.step()