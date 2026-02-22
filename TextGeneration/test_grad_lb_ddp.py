# test_grad_lb_ddp.py
"""
This script analyzes the directional coherence of gradients on a RANDOMLY INITIALIZED
GPT model (i.e., at iteration 0 of training).

It tests the hypothesis that for an untrained model, clean gradients are coherent
while noisy gradients are divergent. This contrasts with analyzing a converged,
pre-trained model.

This modified script also tests the hypothesis that with a larger batch size,
correct gradients accumulate (their summed magnitude is large) while corrupt
gradients cancel out (their summed magnitude is small). It does this by
calculating the L2 norm of the sum of gradients in each category.

It uses torch.func.vmap for efficient per-example gradient computation and is
designed to be run with `torchrun`.
"""
import os
import torch
import numpy as np
import torch.nn.functional as F
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_gather_object, barrier

from torch.func import functional_call, vmap, grad

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# ----- MAIN CONFIGURATION TO MODIFY -----
# --- Model configuration for a new, from-scratch model (GPT-2 124M) ---
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
vocab_size = 50257 # GPT-2 vocab size
bias = True
dropout = 0.0

# Analysis parameters
num_batches_to_test = 200
# Data and Model Config
dataset = 'openwebtext'
wrong_ratio = 0.5 # Using 25% noise to have a good mix

# Batch size for memory-intensive analysis
# Increase batch_size to better observe the accumulation/cancellation effect
# batch_size = 4
batch_size = 8


# System
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # torch.func is not always compatible with compile
backend = 'nccl'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
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
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --- Data Loading Function ---
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

def get_batch_with_wrong_data(split, block_size, ratio):
    data = train_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    change_mask_example = torch.rand(batch_size) < ratio

    for i in range(batch_size):
        if change_mask_example[i]:
            random_values = torch.randint_like(y[i], low=0, high=vocab_size)
            y[i] = random_values

    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y, change_mask_example.to(device)

# --- Initialize a model from scratch instead of loading a checkpoint ---
if master_process:
    print("Initializing a new GPT model from scratch...")
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)
model.eval()

# --- Per-Example Gradient Computation Setup ---
unwrapped_model = model.module if ddp else model
params = dict(unwrapped_model.named_parameters())
buffers = dict(unwrapped_model.named_buffers())

def compute_loss_stateless(params, buffers, x, y):
    batch = (x.unsqueeze(0), y.unsqueeze(0))
    _, loss = functional_call(unwrapped_model, (params, buffers), args=batch)
    return loss

grad_fn = grad(compute_loss_stateless, argnums=0)
per_example_grad_fn = vmap(grad_fn, in_dims=(None, None, 0, 0))

if master_process:
    print("\nStarting DDP Gradient Coherence Analysis on a RANDOMLY INITIALIZED model")
    # Store individual similarity scores from ALL batches and ALL GPUs
    all_clean_vs_clean_sims = []
    all_corrupt_vs_corrupt_sims = []
    all_clean_vs_corrupt_sims = []
    # Store L2 norms of the SUMMED gradients for each batch from ALL GPUs
    all_clean_grad_norms = []
    all_corrupt_grad_norms = []

for i in range(num_batches_to_test):
    X, Y, is_corrupt_mask = get_batch_with_wrong_data('train', unwrapped_model.config.block_size, wrong_ratio)
    
    with ctx:
        per_example_grads_dict = per_example_grad_fn(params, buffers, X, Y)
        grad_key = 'transformer.wte.weight'
        final_layer_grads = per_example_grads_dict[grad_key]

    clean_grads = final_layer_grads[~is_corrupt_mask]
    corrupt_grads = final_layer_grads[is_corrupt_mask]

    # --- L2 NORM CALCULATION (NEW) ---
    # Calculate the L2 norm of the sum of gradients in each category for this GPU's batch.
    # This tests the accumulation (clean) vs. cancellation (corrupt) hypothesis.
    local_clean_norm = 0.0
    if clean_grads.nelement() > 0:
        sum_clean_grads = torch.sum(clean_grads, dim=0)
        local_clean_norm = torch.linalg.norm(sum_clean_grads).item()

    local_corrupt_norm = 0.0
    if corrupt_grads.nelement() > 0:
        sum_corrupt_grads = torch.sum(corrupt_grads, dim=0)
        local_corrupt_norm = torch.linalg.norm(sum_corrupt_grads).item()

    # --- PAIRWISE SIMILARITY CALCULATION (EXISTING) ---
    def calculate_pairwise_similarities(grad_list_1, grad_list_2=None):
        if grad_list_2 is None:
            grad_list_2 = grad_list_1
            is_self_comparison = True
        else:
            is_self_comparison = False

        if len(grad_list_1) < 2 or len(grad_list_2) < 1:
            return []
        
        sims = []
        flat_list_1 = grad_list_1.view(len(grad_list_1), -1)
        flat_list_2 = grad_list_2.view(len(grad_list_2), -1)

        for i in range(len(flat_list_1)):
            start_j = i + 1 if is_self_comparison else 0
            for j in range(start_j, len(flat_list_2)):
                sim = F.cosine_similarity(flat_list_1[i], flat_list_2[j], dim=0)
                sims.append(sim.item())
        
        return sims

    local_cvc_sims = calculate_pairwise_similarities(clean_grads)
    local_pvp_sims = calculate_pairwise_similarities(corrupt_grads)
    local_cvp_sims = calculate_pairwise_similarities(clean_grads, corrupt_grads)

    # --- Gather all results from all GPUs to the master process ---
    if ddp: barrier()
    
    # Prepare containers for gathering
    output_cvc = [None] * ddp_world_size
    output_pvp = [None] * ddp_world_size
    output_cvp = [None] * ddp_world_size
    output_clean_norms = [None] * ddp_world_size
    output_corrupt_norms = [None] * ddp_world_size

    if ddp:
        all_gather_object(output_cvc, local_cvc_sims)
        all_gather_object(output_pvp, local_pvp_sims)
        all_gather_object(output_cvp, local_cvp_sims)
        all_gather_object(output_clean_norms, local_clean_norm)
        all_gather_object(output_corrupt_norms, local_corrupt_norm)
    else: # Handle non-ddp case
        output_cvc = [local_cvc_sims]
        output_pvp = [local_pvp_sims]
        output_cvp = [local_cvp_sims]
        output_clean_norms = [local_clean_norm]
        output_corrupt_norms = [local_corrupt_norm]

    if master_process:
        # Extend master lists with results from all GPUs
        all_clean_vs_clean_sims.extend([item for sublist in output_cvc for item in sublist])
        all_corrupt_vs_corrupt_sims.extend([item for sublist in output_pvp for item in sublist])
        all_clean_vs_corrupt_sims.extend([item for sublist in output_cvp for item in sublist])
        all_clean_grad_norms.extend(output_clean_norms)
        all_corrupt_grad_norms.extend(output_corrupt_norms)
        
        if (i + 1) % 10 == 0:
            print(f"Processed batch {i+1}/{num_batches_to_test}...")

# --- Calculate and report detailed statistics ---
if master_process:
    def report_stats(name, data):
        if not data:
            print(f"\n--- {name} ---")
            print("  (No data to report)")
            return
        
        mean = np.mean(data)
        std = np.std(data)
        mini = np.min(data)
        maxi = np.max(data)
        
        print(f"\n--- {name} ---")
        print(f"  Mean:     {mean:+.6f}")
        print(f"  Std Dev:  {std:.6f}")
        print(f"  Min:      {mini:+.6f}")
        print(f"  Max:      {maxi:+.6f}")

    print("\n" + "="*60)
    print("Gradient Coherence Analysis Complete (Untrained Model)")
    print(f"Averaged over {num_batches_to_test} batches with wrong_ratio={wrong_ratio:.2f}")
    print("-" * 60)
    
    print("Statistics for Pairwise Cosine Similarity:")
    report_stats("Clean vs. Clean Gradients", all_clean_vs_clean_sims)
    report_stats("Corrupt vs. Corrupt Gradients", all_corrupt_vs_corrupt_sims)
    report_stats("Clean vs. Corrupt Gradients", all_clean_vs_corrupt_sims)
    
    print("-" * 60)
    print("Statistics for L2 Norm of Summed Gradients per Batch:")
    report_stats("Sum of Clean Gradients (L2 Norm)", all_clean_grad_norms)
    report_stats("Sum of Corrupt Gradients (L2 Norm)", all_corrupt_grad_norms)

    print("="*60)
    print("\nEXPECTATION (Similarity): Clean vs. Clean should have a positive mean.")
    print("EXPECTATION (Similarity): Other groups should have a mean near zero.")
    print("\nEXPECTATION (L2 Norm): L2 norm of summed CLEAN gradients should be significantly LARGER")
    print("EXPECTATION (L2 Norm): than the L2 norm of summed CORRUPT gradients, showing accumulation vs. cancellation.")


if ddp:
    destroy_process_group()