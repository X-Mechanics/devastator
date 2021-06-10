import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from blur import init_config_run, create_exp_dir, Config, Trainer, ScheduledOptimizer
from blur import Blur, DecoderXL, AdaptiveInput, AdaptiveLogSoftmaxWithLoss
from blur.processing import get_lm_corpus
from blur.training import BalancedDataParallel


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
        default=False,
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

    device = torch.device('cuda' if config_run.cuda and torch.cuda.is_available() else 'cpu')

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = get_lm_corpus(config_run.data, config_run.dataset)
    config_encoder.n_classes = len(corpus.vocab)

    if args.max_step is not None:
        config_run.max_step = args.max_step

    if args.log_interval is not None:
        config_run.log_interval = args.log_interval

    if args.eval_interval is not None:
        config_run.eval_interval = args.eval_interval

    config_run.multi_gpu = args.multi_gpu

    ###############################################################################
    # Build the model
    ###############################################################################

    model = Blur(
        **config_model.parameters(),
        encoder=AdaptiveInput(**config_encoder.parameters()),
        decoder=DecoderXL(**config_decoder.parameters()),
        lm_loss=AdaptiveLogSoftmaxWithLoss(**config_encoder.parameters()),
    )
    model.to(device)

    if config_run.multi_gpu:
        if config_run.gpu0_bsz > 0:
            para_model = BalancedDataParallel(
                config_run.gpu0_bsz // config_run.batch_chunk,
                model, dim=1
            ).to(device)
        else:
            para_model = nn.DataParallel(model)
    else:
        para_model = None

    ###############################################################################
    # Construct optimizer
    ###############################################################################

    optimizer = ScheduledOptimizer(
        config=Config.from_json(config_optim_path),
        model=model
    )
    trainer = Trainer(config_run=config_run, device=device)
    trainer.train(model=model, optimizer=optimizer, corpus=corpus, para_model=para_model)

if __name__ == "__main__":
    main()
