# train_gpt2_100_percent_wrong_12x_batch_wo_add_o.py
# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

import os

wandb_log = True
wandb_project = 'wrong-text-comparison'
wandb_run_name='gpt2-124M-100-percent-wrong-data-woadd-o'
run_id = '100-percent-wrong-data-12x-batch-woadd-o'

# Define the output directory
out_dir = 'result/out_100_percent_12x_batch_woadd_o'

# this makes total number of tokens be 300B
max_iters = 50000
lr_decay_iters = 50000

# low quality data
wrong_ratio = 0.5

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12 * 6
block_size = 1024
gradient_accumulation_steps = 5 * 8 * 2

eval_interval = 100

# eval stuff
# eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1




import os


# Check if the directory exists, and create it if it doesn't
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# Check if the output directory contains a checkpoint file
checkpoint_file = os.path.join(out_dir, 'ckpt.pt')
if os.path.isfile(checkpoint_file):
    init_from = 'resume'
else:
    init_from = 'scratch'


# init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
# init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'


