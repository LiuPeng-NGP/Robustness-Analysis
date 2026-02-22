"""
This script loads a pretrained GPT model from a checkpoint and analyzes the stability 
of its loss on noisy data using Distributed Data Parallel (DDP).

It is designed to be run with `torchrun` to accurately simulate the large, 
distributed batch sizes used during training.

Its purpose is to measure the mean and variance of the loss across multiple
"global macro-batches" to quantify training stability.

How to use:
1.  Place this file in the same directory as train.py, model.py, etc.
2.  Modify the configuration variables in the section below.
3.  Launch with torchrun, just like the training script. Example for 8 GPUs:
    $ torchrun --standalone --nproc_per_node=8 test_loss_ddp.py --batch_size=72 --gradient_accumulation_steps=10
"""
import os
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# ----- MAIN CONFIGURATION TO MODIFY -----
# I/O
out_dir = 'result/out_100_percent_12x_batch_woadd_o' # IMPORTANT: Set this to your checkpoint directory
out_dir = 'result/out_correct' # IMPORTANT: Set this to your checkpoint directory
# Analysis parameters
num_batches_to_test = 200 # How many full "macro-batches" to average over

# Data and Model Config (should match the training run)
dataset = 'openwebtext'
# wrong_ratio = 0.5 # Corresponds to the 100:100 clean:noisy ratio
wrong_ratio = 0

# Batch size configuration (These will be the main variables to change)
# The "global effective batch size" is batch_size * gradient_accumulation_steps * num_gpus
batch_size = 12 * 6 # This is the MICRO-batch size PER GPU
gradient_accumulation_steps = 10 # Accumulation steps PER GPU (originally 5 * 8 * 2 = 80 total, so 10 per GPU)

# System
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
backend = 'nccl'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# --- DDP Initialization ---
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    ddp_world_size = 1
    ddp_rank = 0
    master_process = True
    seed_offset = 0

torch.manual_seed(1337 + seed_offset)
torch.cuda.manual_seed(1337 + seed_offset)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# ---- THIS IS THE CORRECTED LINE ----
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --- Data Loading Function (copied from train.py) ---
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

def get_batch_with_wrong_data(split, block_size, ratio):
    data = train_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    change_mask = torch.rand_like(y, dtype=torch.float32) < ratio
    random_values = torch.randint_like(y, low=0, high=50257)
    y = torch.where(change_mask, random_values, y)
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

# --- Model Loading ---
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
if not os.path.exists(ckpt_path):
    if master_process:
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}.")
    else:
        exit() # Exit silently on other processes

if master_process:
    print(f"Loading model from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)

if compile:
    if master_process: print("Compiling the model...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

model.eval()

# --- Loss Stability Analysis Loop ---
if master_process:
    print("\n" + "="*60)
    print("Starting DDP Loss Stability Analysis")
    print(f"GPUs: {ddp_world_size}")
    print(f"Micro-batch size PER GPU: {batch_size}")
    print(f"Gradient Accumulation Steps PER GPU: {gradient_accumulation_steps}")
    global_effective_batch_size = batch_size * gradient_accumulation_steps * ddp_world_size
    print(f"Global Effective Batch Size: {global_effective_batch_size}")
    print(f"Number of global macro-batches to test: {num_batches_to_test}")
    print("="*60 + "\n")

# This list will store the final, globally-averaged loss for each macro-batch
# It only needs to be populated on the master process
if master_process:
    macro_batch_losses = []

with torch.no_grad():
    for i in range(num_batches_to_test):
        # This will store the loss from each micro-batch on the current rank
        local_micro_losses = []
        for _ in range(gradient_accumulation_steps):
            X, Y = get_batch_with_wrong_data('train', model.module.config.block_size, wrong_ratio)
            with ctx:
                _, loss = model(X, Y)
                # The loss calculated here is ONLY for the data on this GPU
                local_micro_losses.append(loss.item())
        
        # Average the micro-losses on this rank to get the rank's macro-loss
        local_avg_macro_loss = sum(local_micro_losses) / len(local_micro_losses)
        
        # We now need the average loss across ALL GPUs
        loss_tensor = torch.tensor([local_avg_macro_loss], device=device)
        # Sum the loss tensors from all ranks. The result is stored in `loss_tensor` on all ranks.
        all_reduce(loss_tensor, op=ReduceOp.SUM)
        # Divide by the number of GPUs to get the true global average loss
        global_avg_macro_loss = loss_tensor.item() / ddp_world_size
        
        # The master process collects the final global loss for statistics
        if master_process:
            macro_batch_losses.append(global_avg_macro_loss)
            if (i + 1) % 10 == 0:
                print(f"Processed global macro-batch {i+1}/{num_batches_to_test}...")

# --- Calculate and Report Statistics on Master Process ---
if master_process:
    losses_array = np.array(macro_batch_losses)
    mean_loss = np.mean(losses_array)
    variance_loss = np.var(losses_array)
    std_dev_loss = np.std(losses_array)

    print("\n" + "="*50)
    print("Analysis Complete")
    print(f"Tested over {num_batches_to_test} global macro-batches.")
    print(f"Global Effective Batch Size: {global_effective_batch_size}")
    print("-" * 20)
    print(f"Mean of Global Macro-Batch Loss: {mean_loss:.6f}")
    print(f"Variance of Global Macro-Batch Loss: {variance_loss:.6f}")
    print(f"Standard Deviation of Global Macro-Batch Loss: {std_dev_loss:.6f}")
    print("="*50)
    print("\nA lower variance indicates greater training stability.")

if ddp:
    destroy_process_group()