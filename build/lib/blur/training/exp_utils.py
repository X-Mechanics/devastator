import functools
import os, shutil
import time

import torch


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')


def init_config_run(config_run, config_model):
    if not config_model.ext_len >= 0:
        raise ValueError('extended context length must be non-negative')
    if not config_run.batch_size % config_run.batch_chunk == 0:
        raise ValueError('batch chunk does not divide batch size')

    config_run.work_dir = '{}-{}'.format(config_run.work_dir, config_run.dataset)
    config_run.work_dir = os.path.join(config_run.work_dir, time.strftime('%Y%m%d-%H%M%S'))
    config_run.logging = get_logger(log_path=os.path.join(config_run.work_dir, 'log.txt'))
    return config_run


def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)

def create_exp_dir(config_run, scripts_to_save=None):
    dir_path, debug = config_run.work_dir, config_run.debug
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def save_checkpoint(model, optimizer, path, epoch):
    torch.save(model, os.path.join(path, 'model_{}.pt'.format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pt'.format(epoch)))
