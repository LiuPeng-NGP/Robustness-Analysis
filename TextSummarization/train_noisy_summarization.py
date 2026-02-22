# train_summarization_noisy.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tokenizers.implementations import ByteLevelBPETokenizer
from tqdm import tqdm
import logging
import csv
import argparse
from functools import partial
from itertools import islice
# --- MODIFICATION START: Import glob and re for finding checkpoints ---
import glob
import re
# --- MODIFICATION END ---


# DDP (Distributed Data Parallel) Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import datetime

# Import your custom Transformer model
from transformer import Transformer

# --- DDP Setup and Cleanup Functions ---
def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """Destroys the distributed process group."""
    dist.destroy_process_group()

# --- 1. Argument Parsing ---
def parse_args():
    """Parses command-line arguments for noise ratio and total training steps."""
    parser = argparse.ArgumentParser(description="Train a Transformer model for summarization with a specific noise ratio.")
    parser.add_argument(
        "--noise_ratio",
        type=float,
        required=True,
        help="Fraction of noisy data to use for training (a float from 0.0 to 1.0)."
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        required=True,
        help="The total number of training steps for this run."
    )
    args = parser.parse_args()
    if not 0.0 <= args.noise_ratio <= 1.0:
        raise ValueError("noise_ratio must be a float between 0.0 and 1.0.")
    return args

# --- 2. Configuration and Hyperparameters ---
RANK = int(os.environ.get("RANK", 0))

# Training Hyperparameters
# NOTE: BATCH_SIZE is drastically reduced due to MAX_LEN=2048, which requires significant VRAM.
# Adjust this based on your available GPU memory.
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
CLIP = 1.0
EVAL_STEPS = 5000
NUM_EVAL_SAMPLES = 256

# Model Hyperparameters (kept identical to MT for a fair comparison)
VOCAB_SIZE = 32000
MAX_LEN = 2048 # CRITICAL: Increased for long source articles in summarization
D_MODEL = 512
N_LAYERS = 6
N_HEAD = 8
FFN_HIDDEN = 2048
DROP_PROB = 0.1
PAD_IDX = 1
SOS_IDX = 0
EOS_IDX = 2

# --- 3. Data Loading and Collation (Adapted for Summarization) ---
def collate_fn(batch, tokenizer, noise_ratio, is_train):
    """
    Prepares a batch of data for the model.
    Selects between clean and noisy targets based on the noise_ratio.
    """
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(item['article'])
        if is_train:
            # The 'noise_selector' column has values from 0-99.
            # If the selector value is less than the noise percentage, use the noisy target.
            selector = item['noise_selector']
            if selector < (noise_ratio * 100):
                tgt_batch.append(item['highlights_noisy'])
            else:
                tgt_batch.append(item['highlights'])
        else:
            # For validation, always use the clean target.
            tgt_batch.append(item['highlights'])

    # Tokenize batches of source and target texts
    src_tokenized = tokenizer.encode_batch(src_batch)
    tgt_tokenized = tokenizer.encode_batch(tgt_batch)
    
    # Convert tokenized outputs to PyTorch tensors
    src_tensor = torch.tensor([encoding.ids for encoding in src_tokenized], dtype=torch.long)
    tgt_tensor = torch.tensor([encoding.ids for encoding in tgt_tokenized], dtype=torch.long)
    return src_tensor, tgt_tensor

def get_data_loader(tokenizer, batch_size, noise_ratio, dataset_path):
    """Creates training and validation DataLoaders."""
    if not os.path.exists(dataset_path):
        if RANK == 0:
            logging.getLogger(__name__).error(f"Dataset not found at {dataset_path}.")
        exit()
    if RANK == 0:
        logging.getLogger(__name__).info(f"Loading dataset from {dataset_path}")
        
    dataset = load_from_disk(dataset_path)
    
    # Use functools.partial to create collate functions with fixed arguments
    train_collate = partial(collate_fn, tokenizer=tokenizer, noise_ratio=noise_ratio, is_train=True)
    val_collate = partial(collate_fn, tokenizer=tokenizer, noise_ratio=0, is_train=False) # noise_ratio=0 for validation
    
    # Distributed samplers are essential for DDP training
    train_sampler = DistributedSampler(dataset['train'], shuffle=True)
    val_sampler = DistributedSampler(dataset['validation'], shuffle=False, drop_last=True)
    
    train_dataloader = DataLoader(
        dataset['train'], batch_size=batch_size, shuffle=False, collate_fn=train_collate, sampler=train_sampler
    )
    val_dataloader = DataLoader(
        dataset['validation'], batch_size=batch_size, shuffle=False, collate_fn=val_collate, sampler=val_sampler
    )
    return train_dataloader, val_dataloader, train_sampler

# --- 4. Training and Evaluation Functions ---
# --- MODIFICATION START: Added optimizer and paths to function signature for checkpointing ---
def run_evaluation(model, val_dataloader, criterion, device, current_step, best_nll, optimizer, paths, logger):
# --- MODIFICATION END ---
    """
    Runs evaluation, calculates NLL, logs results, and saves the best model.
    """
    model.eval()
    
    # Calculate Negative Log Likelihood on the validation set
    nll_loss = calculate_nll(model, val_dataloader, criterion, device)
    logger.info(f"\tValidation NLL Loss: {nll_loss:.4f} (Best: {best_nll:.4f})")

    # On the main process (rank 0), write results to CSV and save the best model
    if RANK == 0:
        # --- MODIFICATION START: Save a full checkpoint for resuming training ---
        checkpoint_path = os.path.join(paths['model_save_dir'], f"checkpoint_step_{current_step}.pt")
        checkpoint_state = {
            'current_step': current_step,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_nll': best_nll
        }
        torch.save(checkpoint_state, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        # --- MODIFICATION END ---

        with open(paths['csv'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_step, nll_loss])
            
        if nll_loss < best_nll:
            best_nll = nll_loss
            torch.save(model.module.state_dict(), paths['best_model'])
            logger.info(f"🎉 New best model saved with NLL: {nll_loss:.4f} at step {current_step}")

    model.train()
    return best_nll

def train_loop(model, dataloader, optimizer, criterion, device, current_step, train_sampler, best_nll, val_dataloader, tokenizer, paths, logger, total_steps):
    """The main training loop."""
    model.train()
    epoch = 1
    
    while current_step < total_steps:
        if RANK == 0:
            logger.info(f"--- Starting Epoch {epoch} | Current Step: {current_step}/{total_steps} ---")
        train_sampler.set_epoch(epoch)
        
        # Use tqdm for a progress bar, but disable it on non-main processes
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=(RANK != 0))

        for src, trg in pbar:
            if current_step >= total_steps:
                break
                
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            
            # Standard Transformer training step
            output = model(src, trg[:, :-1])
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            current_step += 1

            if RANK == 0:
                pbar.set_postfix({"step": current_step, "loss": loss.item()})
                # Periodically run evaluation based on EVAL_STEPS
                if current_step % EVAL_STEPS == 0:
                    logger.info(f"\n--- Evaluation at step {current_step} ---")
                    # --- MODIFICATION START: Pass optimizer and paths to run_evaluation ---
                    best_nll = run_evaluation(
                        model, val_dataloader, criterion, device,
                        current_step, best_nll, optimizer, paths, logger
                    )
                    # --- MODIFICATION END ---
        
        dist.barrier() # Sync all processes at the end of an epoch
        epoch += 1
    return current_step

@torch.no_grad()
def calculate_nll(model, dataloader, criterion, device):
    """Calculates the average Negative Log Likelihood on a subset of the validation data."""
    model.eval()
    total_loss, num_batches = 0, 0
    
    # Determine how many batches to evaluate to cover NUM_EVAL_SAMPLES
    num_batches_to_eval = (NUM_EVAL_SAMPLES // BATCH_SIZE) // dist.get_world_size()
    val_subset = list(islice(dataloader, num_batches_to_eval))

    for src, trg in val_subset:
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg[:, :-1])
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        total_loss += criterion(output, trg).item()
        num_batches += 1
        
    return total_loss / num_batches if num_batches > 0 else 0.0

# --- MODIFICATION START: Add a function to find the latest checkpoint ---
def find_latest_checkpoint(model_save_dir):
    """Finds the path to the latest checkpoint file based on the step number."""
    checkpoint_files = glob.glob(os.path.join(model_save_dir, "checkpoint_step_*.pt"))
    if not checkpoint_files:
        return None

    latest_step = -1
    latest_checkpoint = None
    for f in checkpoint_files:
        match = re.search(r'checkpoint_step_(\d+).pt', os.path.basename(f))
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_checkpoint = f
    return latest_checkpoint
# --- MODIFICATION END ---

# --- 5. Main Execution ---
def main():
    args = parse_args()
    setup_ddp()
    DEVICE = f"cuda:{os.environ['LOCAL_RANK']}"

    # --- Setup Paths and Naming ---
    MODEL_NAME = f"sum_noise_{args.noise_ratio}"
    RESULTS_DIR = "results"
    MODEL_SAVE_DIR = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_checkpoints")
    
    paths = {
        'log': os.path.join(RESULTS_DIR, f"{MODEL_NAME}_training.log"),
        'csv': os.path.join(RESULTS_DIR, f"{MODEL_NAME}_scores.csv"),
        'best_model': os.path.join(MODEL_SAVE_DIR, "best_model.pt"),
        'model_save_dir': MODEL_SAVE_DIR, # Added for easier access
        'dataset': f"cnn_dailymail_with_noisy_selector_5000k",
        'vocab': "sum-tokenizer/vocab.json",
        'merges': "sum-tokenizer/merges.txt"
    }

    if RANK == 0:
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # --- Setup Logging ---
    logger = logging.getLogger(f"NoiseRatio_{args.noise_ratio}")
    if RANK == 0:
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(paths['log'], mode='a') # Use 'a' to append to log
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    else:
        logger.setLevel(logging.WARNING)

    logger.info(f"[Rank {RANK}] Using device: {DEVICE}")
    logger.info(f"--- Starting Summarization Training with Noise Ratio: {args.noise_ratio} ({args.noise_ratio*100:.1f}%) ---")
    logger.info(f"Total training steps for this run: {args.total_steps}")
    logger.info(f"Effective Batch Size: {BATCH_SIZE * dist.get_world_size()} ({BATCH_SIZE} per GPU)")


    # --- Initialize Tokenizer and Dataloaders ---
    if RANK == 0:
        if not os.path.exists(paths['csv']):
            with open(paths['csv'], 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'nll_loss'])
    dist.barrier()

    tokenizer = ByteLevelBPETokenizer(vocab=paths['vocab'], merges=paths['merges'])
    tokenizer.enable_padding(pad_id=PAD_IDX, pad_token="<pad>", length=MAX_LEN)
    tokenizer.enable_truncation(max_length=MAX_LEN)
    
    train_dl, val_dl, train_sampler = get_data_loader(tokenizer, BATCH_SIZE, args.noise_ratio, paths['dataset'])
    if RANK == 0:
        logger.info("DataLoaders created.")

    # --- Initialize Model, Optimizer, and Loss Function ---
    model = Transformer(
        src_pad_idx=PAD_IDX, trg_pad_idx=PAD_IDX, trg_sos_idx=SOS_IDX,
        enc_voc_size=VOCAB_SIZE, dec_voc_size=VOCAB_SIZE, d_model=D_MODEL,
        n_head=N_HEAD, max_len=MAX_LEN, ffn_hidden=FFN_HIDDEN,
        n_layers=N_LAYERS, drop_prob=DROP_PROB, device=DEVICE,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # --- MODIFICATION START: Logic to load the latest checkpoint ---
    current_step = 0
    best_nll = float('inf')
    
    latest_checkpoint_path = find_latest_checkpoint(MODEL_SAVE_DIR)
    if latest_checkpoint_path:
        if RANK == 0:
            logger.info(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        # Map location to current device to avoid CUDA errors
        checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE)
        
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_step = checkpoint['current_step']
        best_nll = checkpoint['best_nll']
        
        if RANK == 0:
            logger.info(f"Resumed from step {current_step} with best NLL {best_nll:.4f}")
    else:
        if RANK == 0:
            logger.info(f"--- Starting fresh training up to {args.total_steps} steps ---")
    # --- MODIFICATION END ---
    
    # --- Start Training ---
    train_loop(
        model, train_dl, optimizer, criterion, DEVICE, current_step,
        train_sampler, best_nll, val_dl, tokenizer, paths, logger,
        total_steps=args.total_steps
    )
    
    if RANK == 0:
        logger.info("--- Training Finished ---")
    cleanup_ddp()

if __name__ == "__main__":
    def generate(self, src, max_len, sos_idx, eos_idx):
        self.eval()
        batch_size = src.shape[0]
        trg = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=self.device)
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        
        for _ in range(max_len - 1):
            trg_mask = self.make_trg_mask(trg)
            output = self.decoder(trg, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(2)[:, -1].unsqueeze(1)
            trg = torch.cat((trg, pred_token), dim=1)
            if (trg == eos_idx).any(dim=-1).sum() == batch_size:
                break
        return trg
    Transformer.generate = generate
    
    main()