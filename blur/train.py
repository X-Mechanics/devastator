import argparse
import time
import math
import os
import itertools
import dataclasses
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from configs.xlmodelconfig import XlModelConfig
from configs.fnetarmodelconfig import FnetarModelConfig
from configs.feedbackmodelconfig import FeedbackModelConfig

from configs.xladaptiveconfig import XlAdaptiveConfig
from configs.feedbackadaptiveconfig import FeedbackAdaptiveConfig

from configs.xldataconfig import XlDataConfig
from configs.feedbackdataconfig import FeedbackDataConfig

from configs.runconfig import RunConfig
from configs.optimizerconfig import OptimizerConfig

from blur import Blur

from models.xl import Xl
from models.fnetar import Fnetar
from models.feedback import Feedback

from modules.xlmemories import XlMemories
from modules.feedbackmemories import FeedbackMemories

from modules.adaptiveinput import AdaptiveInput
from modules.adaptivelogsoftmax import AdaptiveLogSoftmax

from utils.data_utils import get_lm_corpus
from utils.exp_utils import create_exp_dir

from models.utils.normaluniforminitializer import NormalUniformInitializer

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--model_name', type=str, choices=['xl', 'fnetar', 'feedback'], help='optimizer to use.')
parser.add_argument('--dataset', default='wt103', type=str, help='data')
parser.add_argument('--data', default='../data/wikitext-103', type=str, help='dataset')

args = parser.parse_args()
print('Training new {} model'.format(args.model_name))

run_config = RunConfig()
optimizer_config = OptimizerConfig()

run_config.work_dir = os.path.join(run_config.work_dir, time.strftime('%Y%m%d-%H%M%S'))
logging = create_exp_dir(run_config.work_dir, scripts_to_save=['train.py', 'blur.py'], debug=run_config.debug)

# Set the random seed manually for reproducibility.
np.random.seed(run_config.seed)
torch.manual_seed(run_config.seed)

if torch.cuda.is_available():
    if not run_config.cuda:
        device = torch.device('cpu')
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(run_config.seed)
else:
    device = torch.device('cpu')

corpus = get_lm_corpus(args.data, args.dataset)

if args.model_name == 'xl':
    data_config = XlDataConfig()
    adaptive_config = XlAdaptiveConfig(n_classes=len(corpus.vocab))
    model_config = XlModelConfig()
    transformer = Xl(**dataclasses.asdict(model_config))
elif args.model_name == 'fnetar':
    data_config = XlDataConfig()
    adaptive_config = XlAdaptiveConfig(n_classes=len(corpus.vocab))
    model_config = FnetarModelConfig()
    transformer = Fnetar(**dataclasses.asdict(model_config))
elif args.model_name == 'feedback':
    data_config = FeedbackDataConfig()
    adaptive_config = FeedbackAdaptiveConfig(n_classes=len(corpus.vocab))
    model_config = FeedbackModelConfig()
    transformer = Feedback(**dataclasses.asdict(model_config))
else:
    raise ValueError

encoder = AdaptiveInput(**dataclasses.asdict(adaptive_config))
decoder = AdaptiveLogSoftmax(**dataclasses.asdict(adaptive_config))
model = Blur(encoder=encoder, transformer=transformer, decoder=decoder, tie_weight=True)

###############################################################################
# Load data
###############################################################################

assert data_config.batch_size % data_config.batch_chunk == 0

tr_iter = corpus.get_iterator('train', data_config.batch_size, data_config.tgt_len,
    device=device, ext_len=0)
va_iter = corpus.get_iterator('valid', data_config.eval_batch_size, data_config.eval_tgt_len,
    device=device, ext_len=0)
te_iter = corpus.get_iterator('test', data_config.eval_batch_size, data_config.eval_tgt_len,
    device=device, ext_len=0)


###############################################################################
# Build the model
###############################################################################

initializer = NormalUniformInitializer()
model.apply(initializer)
model.encoder.apply(initializer) # ensure embedding init is not overridden by out_layer in case of weight sharing

args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.transformer.parameters()])
args.n_encoder_param = sum([p.nelement() for p in model.encoder.parameters()])
args.n_decoder_param = sum([p.nelement() for p in model.decoder.parameters()])

para_model = model.to(device)

#### optimizer
optimizer = optim.Adam(model.parameters(), lr=optimizer_config.lr)

#### scheduler
# here we do not set eta_min to lr_min to be backward compatible
# because in previous versions eta_min is default to 0
# rather than the default value of lr_min 1e-6
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
    optimizer_config.max_step, eta_min=optimizer_config.eta_min) # should use eta_min arg

logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))
logging('#encoder params = {}'.format(args.n_encoder_param))
logging('#decoder params = {}'.format(args.n_decoder_param))

###############################################################################
# Training code
###############################################################################

def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # Evaluation
    total_len, total_loss = 0, 0.

    if args.model_name == 'xl' or args.model_name == 'fnetar':
        eval_memories = XlMemories(
            n_stream=1,
            n_layer=data_config.n_layer,
            tgt_len=data_config.eval_tgt_len,
            mem_len=data_config.eval_mem_len,
            ext_len=0,
            dtype=next(model.parameters()).dtype
        )
    else:
        eval_memories = FeedbackMemories(n_stream=1)


    with torch.no_grad():

        for i, (data, target, seq_len) in enumerate(eval_iter):
            if run_config.max_eval_steps > 0 and i >= run_config.max_eval_steps:
                break
            loss, new_eval_memory = model(data, target, eval_memories[0])
            eval_memories.update_memory_stream(stream_index=0, memory=new_eval_memory)

            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    # Switch back to the training mode
    model.train()

    return total_loss / total_len


def train():
    # Turn on training mode which enables dropout.
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    model.train()


    if args.model_name == 'xl' or args.model_name == 'fnetar':
        memories = XlMemories(
            n_stream=data_config.batch_chunk,
            n_layer=data_config.n_layer,
            tgt_len=data_config.tgt_len,
            mem_len=data_config.mem_len,
            ext_len=0,
            dtype=next(model.parameters()).dtype
        )
    else:
        memories = FeedbackMemories(n_stream=data_config.batch_chunk)

    train_iter = tr_iter
    for batch, (data, target, seq_len) in tqdm(enumerate(train_iter), total=len(train_iter) // data_config.batch_chunk):
        model.zero_grad()

        data_chunks = torch.chunk(data, data_config.batch_chunk, 0)
        target_chunks = torch.chunk(target, data_config.batch_chunk, 0)
        for i in range(data_config.batch_chunk):
            data_i = data_chunks[i]
            target_i = target_chunks[i]
            memory_i = memories[i]
            loss, new_memory_i = para_model(data_i, target_i, memory_i)
            memories.update_memory_stream(stream_index=i, memory=new_memory_i)

            loss = loss.float().mean().type_as(loss) / data_config.batch_chunk
            loss.backward()
            train_loss += loss.float().item()


        torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_config.clip)
        optimizer.step()

        # step-wise learning rate annealing
        train_step += 1

        # linear warmup stage
        if train_step < optimizer_config.warmup_step:
            curr_lr = optimizer_config.lr * train_step / optimizer_config.warmup_step
            optimizer.param_groups[0]['lr'] = curr_lr

        else:
            scheduler.step()


        if train_step % run_config.log_interval == 0:
            cur_loss = train_loss / run_config.log_interval
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                      '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, train_step, batch+1, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / run_config.log_interval, cur_loss)
            log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
            logging(log_str)
            train_loss = 0
            log_start_time = time.time()

        if train_step % run_config.eval_interval == 0:
            val_loss = evaluate(va_iter)
            logging('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.2f}'.format(
                train_step // run_config.eval_interval, train_step,
                (time.time() - eval_start_time), val_loss)
            log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
            logging(log_str)
            logging('-' * 100)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if not run_config.debug:
                    with open(os.path.join(run_config.work_dir, 'model.pt'), 'wb') as f:
                        torch.save(model, f)
                    with open(os.path.join(run_config.work_dir, 'optimizer.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)
                best_val_loss = val_loss

            eval_start_time = time.time()

        if train_step == optimizer_config.max_step:
            break

# Loop over epochs.
train_step = 0
train_loss = 0
best_val_loss = None

log_start_time = time.time()
eval_start_time = time.time()

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in itertools.count(start=1):
        train()
        if train_step == optimizer_config.max_step:
            logging('-' * 100)
            logging('End of training')
            break
except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')

# Load the best saved model.
with open(os.path.join(run_config.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
para_model = model.to(device)

# Run on test data.
test_loss = evaluate(te_iter)
logging('=' * 100)

logging('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
    test_loss, math.exp(test_loss)))
logging('=' * 100)
