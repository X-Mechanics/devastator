import time
import math
import os
import itertools
import torch


###############################################################################
# Training code
###############################################################################

from tqdm import tqdm

class Trainer:
    def __init__(self, config_run, device):
        self.config_run = config_run
        self.device = device

    def _init_params(self):
        self.epoch = 1
        self.train_step = 0
        self.train_loss = 0
        self.best_val_loss = None
        self.log_start_time = time.time()
        self.eval_start_time = time.time()

    def get_train_iter(self, corpus, model):
        return corpus.get_iterator(
            'train', self.config_run.batch_size, model.tgt_len,
            device=self.device, ext_len=model.ext_len
        )

    def get_val_iter(self, corpus, model):
        return corpus.get_iterator(
            'valid', self.config_run.eval_batch_size, self.config_run.eval_tgt_len,
            device=self.device, ext_len=model.ext_len
        )

    def train(self, model, optimizer, corpus, para_model=None):
        if para_model is None:
            para_model = model

        tr_iter = self.get_train_iter(corpus, model)
        va_iter = self.get_val_iter(corpus, model)

        self._init_params()

        for epoch in itertools.count(start=1):
            self.epoch = epoch

            self.config_run.logging('training epoch {}...'.format(epoch))
            ret = self.train_epoch(model, optimizer, tr_iter, va_iter, para_model=para_model)

            if not ret:
                break

    def train_epoch(self, model, optimizer, tr_iter, va_iter, para_model=None):
        ret = True
        model.train()

        if self.config_run.batch_chunk > 1:
            mems = [tuple() for _ in range(self.config_run.batch_chunk)]
        else:
            mems = tuple()

        train_iter = tr_iter.get_varlen_iter() if self.config_run.varlen else tr_iter
        batches_per_epoch = int((tr_iter.data.size(0) - 1)/tr_iter.bptt)

        for batch, (data, target, seq_len) in enumerate(tqdm(train_iter, total=batches_per_epoch)):
            model.zero_grad()

            if self.config_run.batch_chunk > 1:
                data_chunks = torch.chunk(data, self.config_run.batch_chunk, 1)
                target_chunks = torch.chunk(target, self.config_run.batch_chunk, 1)

                for i in range(self.config_run.batch_chunk):
                    data_i = data_chunks[i].contiguous()
                    target_i = target_chunks[i].contiguous()

                    output = para_model(data_i, target_i, mems[i])

                    mems[i] = output['mems']
                    loss = output['loss'].mean() / self.config_run.batch_chunk

                    if self.config_run.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()

                    self.train_loss += loss.cpu().detach().item()
            else:
                output = para_model(data, target, mems)
                loss = output['loss'].mean()
                mems = output['mems']

                if self.config_run.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                self.train_loss += loss.cpu().detach().item()

            if self.config_run.fp16:
                optimizer.clip_master_grads(self.config_run.clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config_run.clip)

            optimizer.step()

            # step-wise learning rate annealing
            self.train_step += 1
            optimizer.update(self.train_step)

            if self.train_step % self.config_run.log_interval == 0:
                cur_loss = self.train_loss / self.config_run.log_interval
                elapsed = time.time() - self.log_start_time
                log_str = '| epoch {:3d} step {:>8d} | lr {:.3g} ' \
                          '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                    self.epoch, self.train_step, optimizer.optim.param_groups[0]['lr'],
                                       elapsed * 1000 / self.config_run.log_interval, cur_loss)
                log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
                self.config_run.logging(log_str)
                self.train_loss = 0
                self.log_start_time = time.time()

            if self.train_step % self.config_run.eval_interval == 0 and va_iter is not None:

                val_loss = self.run_eval(model=model, eval_iter=va_iter)
                self.config_run.logging('-' * 100)
                log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                          '| valid loss {:5.2f}'.format(
                    self.train_step // self.config_run.eval_interval, self.train_step,
                    (time.time() - self.eval_start_time), val_loss)
                log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
                self.config_run.logging(log_str)
                self.config_run.logging('-' * 100)

                # Save the model if the validation loss is the best we've seen so far.
                if not self.best_val_loss or val_loss < self.best_val_loss:
                    if not self.config_run.debug:
                        with open(os.path.join(self.config_run.work_dir, 'model.pt'), 'wb') as f:
                            torch.save(model, f)
                        with open(os.path.join(self.config_run.work_dir, 'optimizer.pt'), 'wb') as f:
                            torch.save(optimizer.optim.state_dict(), f)
                    self.best_val_loss = val_loss

                self.eval_start_time = time.time()

            if not self.train_loss==self.train_loss:
                raise RuntimeError('NaN values have entered the base!')

            if self.train_step > self.config_run.max_step:
                ret = False
                break
        return ret

    def run_eval(self, model, eval_iter):
        # Turn on evaluation mode which disables dropout.
        model.eval()

        tgt_len, mem_len, ext_len = model.tgt_len, model.mem_len, model.ext_len

        # If model has no memory make ext_len longer, else make mem_len longer and keep ext_len same
        if mem_len == 0:
            model._reset_length(self.config_run.eval_tgt_len,
                                ext_len + tgt_len - self.config_run.eval_tgt_len, mem_len)
        else:
            model._reset_length(self.config_run.eval_tgt_len,
                                ext_len, mem_len + tgt_len - self.config_run.eval_tgt_len)

        # Evaluation
        total_len, total_loss = 0, 0.
        with torch.no_grad():
            mems = tuple()
            for i, (data, target, seq_len) in enumerate(eval_iter):
                if self.config_run.max_eval_steps > 0 and i >= self.config_run.max_eval_steps:
                    break
                output = model(data, target, mems)
                loss = output['loss'].mean()
                mems = output['mems']
                total_loss += seq_len * loss.cpu().detach().item()
                total_len += seq_len

        # Switch back to the training mode
        model._reset_length(tgt_len, ext_len, mem_len)
        model.train()

        return total_loss / total_len