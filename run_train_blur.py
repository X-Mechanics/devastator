import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from blur import init_config_run, create_exp_dir, Config, Trainer, ScheduledOptimizer
from blur import Blur, DecoderXL, AdaptiveInput, AdaptiveLogSoftmaxWithLoss
from blur.processing import get_lm_corpus
from blur.training.stream_dataset import StreamDataset, StreamCollator

import time
import math
import itertools
from tqdm import tqdm
import datetime

def train(gpu, args, config_run, config_model, config_encoder, config_decoder, config_optim):
    device = torch.device('cuda' if config_run.cuda and torch.cuda.is_available() else 'cpu')

    corpus = get_lm_corpus(config_run.data, config_run.dataset)
    config_encoder.n_classes = len(corpus.vocab)

    model = Blur(
        **config_model.parameters(),
        encoder=AdaptiveInput(**config_encoder.parameters()),
        decoder=DecoderXL(**config_decoder.parameters()),
        lm_loss=AdaptiveLogSoftmaxWithLoss(**config_encoder.parameters()),
    )
    model.to(device)

    train_set = StreamDataset(
        data=corpus.train, tgt_len=model.tgt_len, batch_size=config_run.batch_size)
    valid_set = StreamDataset(
        data=corpus.valid, tgt_len=config_run.eval_tgt_len, batch_size=config_run.batch_size)

    if config_run.multi_gpu:
        import torch.distributed as dist

        args.world_size = args.gpus * args.nodes  #
        os.environ['MASTER_ADDR'] = 'localhost'  #
        os.environ['MASTER_PORT'] = '8700'  #

        rank = args.nr * args.gpus + gpu
        print('initializing disitrbuted training...', args.world_size, rank)
        dist.init_process_group(
            backend='nccl', init_method='env://', world_size=args.world_size, rank=rank,
            timeout=datetime.timedelta(0, 30)
        )
        print('finished initializing...')
        mod = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=args.world_size, rank=rank)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_set, num_replicas=args.world_size, rank=rank)
    else:
        mod = model
        train_sampler, valid_sampler = None, None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=config_run.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=train_sampler, collate_fn=StreamCollator())
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_set, batch_size=config_run.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=valid_sampler, collate_fn=StreamCollator())

    optimizer = ScheduledOptimizer(config=config_optim, model=model)

    train_step = 0
    train_loss = 0
    best_val_loss = None
    log_start_time = time.time()
    eval_start_time = time.time()

    for epoch in itertools.count(start=1):
        epoch = epoch

        config_run.logging('training epoch {}...'.format(epoch))

        ret = True
        model.train()

        mems = tuple()

        for batch, (data, target) in enumerate(tqdm(train_loader)):
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            model.zero_grad()

            output = mod(data, target, mems)
            loss = output['loss'].mean()
            mems = output['mems']

            if config_run.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            train_loss += loss.cpu().detach().item()

            if config_run.fp16:
                optimizer.clip_master_grads(config_run.clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config_run.clip)

            optimizer.step()

            # step-wise learning rate annealing
            train_step += 1
            optimizer.update(train_step)

            if train_step % config_run.log_interval == 0:
                cur_loss = train_loss / config_run.log_interval
                elapsed = time.time() - log_start_time
                log_str = '| epoch {:3d} step {:>8d} | lr {:.3g} | ms/batch {:5.2f} | loss {:5.2f}'.format(
                    epoch, train_step, optimizer.optim.param_groups[0]['lr'],
                    elapsed * 1000 / config_run.log_interval, cur_loss)
                log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
                config_run.logging(log_str)
                train_loss = 0
                log_start_time = time.time()


            if train_step % config_run.eval_interval == 0:
                model.eval()

                tgt_len, mem_len, ext_len = model.tgt_len, model.mem_len, model.ext_len

                # If model has no memory make ext_len longer, else make mem_len longer and keep ext_len same
                if mem_len == 0:
                    model._reset_length(
                        config_run.eval_tgt_len, ext_len + tgt_len - config_run.eval_tgt_len, mem_len)
                else:
                    model._reset_length(
                        config_run.eval_tgt_len, ext_len, mem_len + tgt_len - config_run.eval_tgt_len)

                # Evaluation
                total_len, total_loss = 0, 0.
                with torch.no_grad():
                    mems = tuple()
                    for i, (data, target) in enumerate(valid_loader):
                        data = data.to(device)
                        target  = target.to(device)

                        if config_run.max_eval_steps > 0 and i >= config_run.max_eval_steps:
                            break
                        output = mod(data, target, mems)
                        loss = output['loss'].mean()
                        mems = output['mems']
                        seq_len = model.tgt_len
                        total_loss += seq_len * loss.cpu().detach().item()
                        total_len += seq_len

                # Switch back to the training mode
                model._reset_length(tgt_len, ext_len, mem_len)
                model.train()

                val_loss = total_loss / total_len


                config_run.logging('-' * 100)
                log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s | valid loss {:5.2f}'.format(
                    train_step // config_run.eval_interval, train_step,
                    (time.time() - eval_start_time), val_loss)
                log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
                config_run.logging(log_str)
                config_run.logging('-' * 100)

                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    if not config_run.debug:
                        with open(os.path.join(config_run.work_dir, 'model.pt'), 'wb') as f:
                            torch.save(model, f)
                        with open(os.path.join(config_run.work_dir, 'optimizer.pt'), 'wb') as f:
                            torch.save(optimizer.optim.state_dict(), f)
                    best_val_loss = val_loss

                eval_start_time = time.time()

            if not train_loss == train_loss:
                raise RuntimeError('NaN values have entered the base!')

            if train_step > config_run.max_step:
                ret = False
                break

        if not ret:
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default='test_data/blur/wikitext-103',
        type=str,
        help="Base data directory",
    )

    parser.add_argument(
        "--dataset",
        default='wt103',
        type=str,
        help="dataset name",
    )

    parser.add_argument(
        "--max_step",
        default=None,
        type=int,
        help="Number of training steps until stop",
    )

    parser.add_argument(
        "--log_interval",
        default=None,
        type=int,
        help="Number of training steps between logging",
    )

    parser.add_argument(
        "--eval_interval",
        default=None,
        type=int,
        help="Number of training steps between eval",
    )

    parser.add_argument(
        "--multi_gpu",
        default=True,
        type=bool,
        help="Option use nn.DataParallel",
    )

    args = parser.parse_args()

    ###############################################################################
    # Define config paths
    ###############################################################################

    config_dir = os.path.join('configs', 'xl')
    config_run_path = os.path.join(config_dir, 'config_run.json')
    config_encoder_path = os.path.join(config_dir, 'config_encoder.json')
    config_decoder_path = os.path.join(config_dir, 'config_decoder.json')
    config_optim_path = os.path.join(config_dir, 'config_optim.json')

    ###############################################################################
    # Load config files
    ###############################################################################

    config_model = Config(**{"tgt_len": 150, "mem_len": 150, "ext_len": 0})
    config_run = init_config_run(config_run=Config.from_json(config_run_path), config_model=config_model)
    config_encoder = Config.from_json(config_encoder_path)
    config_decoder = Config.from_json(config_decoder_path)
    config_optim = Config.from_json(config_optim_path)

    ###############################################################################
    # Setup training
    ###############################################################################

    create_exp_dir(config_run, scripts_to_save=['blur/modeling/blur.py'])

    np.random.seed(config_run.seed);  # Set the random seed manually
    torch.manual_seed(config_run.seed);

    if torch.cuda.is_available():  # Set hardware
        if not config_run.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed_all(config_run.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    if args.max_step is not None:
        config_run.max_step = args.max_step

    if args.log_interval is not None:
        config_run.log_interval = args.log_interval

    if args.eval_interval is not None:
        config_run.eval_interval = args.eval_interval

    args.nodes = 1
    args.gpus = 1
    args.nr = 0

    config_run.multi_gpu = args.multi_gpu

    ###############################################################################
    # Build the model
    ###############################################################################

    # train(-1, args, config_run, config_model, config_encoder, config_decoder, config_optim)
    mp.spawn(
        train, nprocs=args.gpus,
        args=(args, config_run, config_model, config_encoder, config_decoder, config_optim))

if __name__ == "__main__":
    main()
