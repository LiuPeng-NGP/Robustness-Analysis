# train_correct.py
# for text summarization

# train_summarization.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tokenizers.implementations import ByteLevelBPETokenizer
from tqdm import tqdm
import logging
from itertools import islice
import csv
import numpy as np

# DDP (Distributed Data Parallel) Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import datetime

# --- Metric Imports ---
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calc

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


# --- 1. Configuration and Hyperparameters ---
RANK = int(os.environ.get("RANK", 0))

# Paths & Naming
DATASET_PATH = "cnn_dailymail"
VOCAB_FILE = "sum-tokenizer/vocab.json"
MERGES_FILE = "sum-tokenizer/merges.txt"
RESULTS_DIR = "results"
MODEL_NAME = "sum_clean_teacher"

# Define the model save directory to be inside the results directory
MODEL_SAVE_DIR = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_checkpoints")

# Create directories if they don't exist
if RANK == 0:
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

# File paths for logging and results
LOG_FILE = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_training.log")
RESULTS_CSV_FILE = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_scores.csv")
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pt")
CHECKPOINT_FILENAME_TEMPLATE = "model_step_{}.pt"

# Setup logging (only on the main process to avoid clutter)
logger = logging.getLogger(__name__)
if RANK == 0:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_FILE, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
else:
    logger.setLevel(logging.WARNING)


# Training Hyperparameters
# NOTE: BATCH_SIZE is significantly smaller than MT due to the larger MAX_LEN,
# which greatly increases memory consumption. Adjust based on available VRAM.
BATCH_SIZE = 16  # Per GPU
LEARNING_RATE = 1e-4
CLIP = 1.0

# Step-based Training Configuration
TOTAL_TRAIN_STEPS = 50000
EVAL_SAVE_STEPS = 5000
NUM_EVAL_SAMPLES = 256

# Model Hyperparameters
VOCAB_SIZE = 32000
MAX_LEN = 2048 # Significantly larger for summarization articles
D_MODEL = 512
N_LAYERS = 6
N_HEAD = 8
FFN_HIDDEN = 2048
DROP_PROB = 0.1

# Tokenizer Special IDs
PAD_IDX = 1
SOS_IDX = 0
EOS_IDX = 2


# --- 2. Data Loading and Collation (Adapted for Summarization) ---
def collate_fn(batch, tokenizer):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(item['article'])
        tgt_batch.append(item['highlights'])
    src_tokenized = tokenizer.encode_batch(src_batch)
    tgt_tokenized = tokenizer.encode_batch(tgt_batch)
    src_tensor = torch.tensor([encoding.ids for encoding in src_tokenized], dtype=torch.long)
    tgt_tensor = torch.tensor([encoding.ids for encoding in tgt_tokenized], dtype=torch.long)
    return src_tensor, tgt_tensor

def get_data_loader(tokenizer, batch_size):
    if not os.path.exists(DATASET_PATH):
        if RANK == 0: logger.error(f"Dataset not found at {DATASET_PATH}.")
        exit()
    if RANK == 0: logger.info(f"Loading dataset from {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)
    collate_wrapper = lambda batch: collate_fn(batch, tokenizer)
    train_sampler = DistributedSampler(dataset['train'], shuffle=True)
    val_sampler = DistributedSampler(dataset['validation'], shuffle=False, drop_last=True)
    train_dataloader = DataLoader(
        dataset['train'], batch_size=batch_size, shuffle=False,
        collate_fn=collate_wrapper, sampler=train_sampler
    )
    val_dataloader = DataLoader(
        dataset['validation'], batch_size=batch_size, shuffle=False,
        collate_fn=collate_wrapper, sampler=val_sampler
    )
    return train_dataloader, val_dataloader, train_sampler


# --- 3. Training and Evaluation Functions (Adapted for Summarization) ---

def run_evaluation(model, val_dataloader, criterion, tokenizer, device, current_step, best_bert_f1):
    """Orchestrates the evaluation process for summarization."""
    model.eval()

    nll_loss = calculate_nll(model, val_dataloader, criterion, device)
    logger.info(f"\tValidation NLL Loss: {nll_loss:.4f}")

    rouge_l, bert_f1 = generate_and_score(model, val_dataloader, tokenizer, device)
    logger.info(f"\tValidation ROUGE-L F1: {rouge_l:.4f}")
    logger.info(f"\tValidation BERTScore F1: {bert_f1:.4f}")

    with open(RESULTS_CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([current_step, rouge_l, bert_f1, nll_loss])

    if bert_f1 > best_bert_f1:
        best_bert_f1 = bert_f1
        torch.save(model.module.state_dict(), BEST_MODEL_PATH)
        logger.info(f"🎉 New best model saved with BERTScore F1: {bert_f1:.4f} at step {current_step}")

    model.train()
    return best_bert_f1

def train_loop(model, dataloader, optimizer, criterion, clip, device, current_step, train_sampler, best_bert_f1, val_dataloader, tokenizer):
    model.train()
    epoch = 1
    
    while current_step < TOTAL_TRAIN_STEPS:
        if RANK == 0:
            logger.info(f"--- Starting Epoch {epoch} | Current Step: {current_step} ---")
        train_sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=(RANK != 0))

        for src, trg in pbar:
            if current_step >= TOTAL_TRAIN_STEPS:
                break

            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            current_step += 1

            if RANK == 0:
                pbar.set_postfix({"step": current_step, "loss": loss.item()})
                if current_step % EVAL_SAVE_STEPS == 0:
                    logger.info(f"\n--- Evaluation at step {current_step} ---")
                    
                    checkpoint_path = os.path.join(MODEL_SAVE_DIR, CHECKPOINT_FILENAME_TEMPLATE.format(current_step))
                    save_payload = {'step': current_step, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                    torch.save(save_payload, checkpoint_path)
                    logger.info(f"✅ Checkpoint saved to {checkpoint_path}")

                    best_bert_f1 = run_evaluation(model, val_dataloader, criterion, tokenizer, device, current_step, best_bert_f1)
        
        dist.barrier()
        epoch += 1
    return current_step

@torch.no_grad()
def calculate_nll(model, dataloader, criterion, device):
    """Calculates the Negative Log Likelihood on the validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    num_batches_to_eval = (NUM_EVAL_SAMPLES // BATCH_SIZE) // dist.get_world_size()
    val_subset = list(islice(dataloader, num_batches_to_eval))

    pbar = tqdm(val_subset, desc="Calculating NLL", disable=(RANK != 0))
    for src, trg in pbar:
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg[:, :-1])
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0

@torch.no_grad()
def generate_and_score(model, dataloader, tokenizer, device):
    """Generates summaries and calculates ROUGE and BERTScore."""
    model.eval()
    candidate_strings, reference_strings = [], []
    num_batches_to_eval = (NUM_EVAL_SAMPLES // BATCH_SIZE) // dist.get_world_size()
    val_subset = list(islice(dataloader, num_batches_to_eval))

    pbar = tqdm(val_subset, desc="Generating summaries for scoring", disable=(RANK != 0))
    for src, trg in pbar:
        src = src.to(device)
        
        # --- MODIFICATION START ---
        # BUG FIX: The previous implementation used a loop of `tokenizer.decode`,
        # which is inefficient and can lead to errors in handling special tokens.
        # The corrected approach uses `decode_batch`, which is the standard,
        # robust way to perform this operation. This is the critical fix for
        # the near-zero ROUGE scores.

        # Decode the batch of reference summaries from the dataloader tensors
        reference_strings.extend(
            tokenizer.decode_batch(trg.cpu().numpy(), skip_special_tokens=True)
        )
        
        # Generate summaries from the source articles
        # NOTE: For summaries, the max_len of the output can be smaller than the input.
        # We cap it at a reasonable value like 256 to speed up generation.
        batch_summaries = model.module.generate(src, max_len=256, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
        
        # Decode the batch of generated summaries
        candidate_strings.extend(
            tokenizer.decode_batch(batch_summaries.cpu().numpy(), skip_special_tokens=True)
        )
        # --- MODIFICATION END ---

    if not candidate_strings or not reference_strings: return 0.0, 0.0

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_l_f1_scores = []
    for ref, cand in zip(reference_strings, candidate_strings):
        # Add a check for empty candidates, which can result from decoding errors
        if not cand.strip(): 
            rouge_l_f1_scores.append(0.0)
            continue
        scores = scorer.score(ref, cand)
        rouge_l_f1_scores.append(scores['rougeL'].fmeasure)
    avg_rouge_l_f1 = np.mean(rouge_l_f1_scores) if rouge_l_f1_scores else 0.0

    # Calculate BERTScore
    # Filter out empty strings to avoid errors in BERTScore calculation
    filtered_cands = [c for c, r in zip(candidate_strings, reference_strings) if c.strip() and r.strip()]
    filtered_refs = [r for c, r in zip(candidate_strings, reference_strings) if c.strip() and r.strip()]

    if not filtered_cands or not filtered_refs:
        return avg_rouge_l_f1, 0.0

    P, R, F1 = bert_score_calc(filtered_cands, filtered_refs, lang="en", verbose=(RANK==0))
    
    return avg_rouge_l_f1, F1.mean().item()


# --- 4. Main Execution ---
def main():
    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{local_rank}"
    logger.info(f"[Rank {RANK}] Using device: {DEVICE}")

    if RANK == 0:
        logger.info("--- Pre-caching models on main process (rank 0) ---")
        tokenizer = ByteLevelBPETokenizer(vocab=VOCAB_FILE, merges=MERGES_FILE)
        logger.info("Tokenizer loaded.")
        bert_score_calc(["."],[","], lang="en", verbose=False) # Use 'en' for summarization
        logger.info("BERTScore model cached. Pre-caching complete.")
        if not os.path.exists(RESULTS_CSV_FILE):
            with open(RESULTS_CSV_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'rouge_l', 'bertscore_f1', 'nll_loss']) # Adapted header
    dist.barrier()

    tokenizer = ByteLevelBPETokenizer(vocab=VOCAB_FILE, merges=MERGES_FILE)
    # The tokenizer padding/truncation length must match the model's max_len
    tokenizer.enable_padding(pad_id=PAD_IDX, pad_token="<pad>", length=MAX_LEN)
    tokenizer.enable_truncation(max_length=MAX_LEN)
    
    train_dataloader, val_dataloader, train_sampler = get_data_loader(tokenizer, BATCH_SIZE)
    if RANK == 0: logger.info("DataLoaders created.")

    model = Transformer(
        src_pad_idx=PAD_IDX, trg_pad_idx=PAD_IDX, trg_sos_idx=SOS_IDX,
        enc_voc_size=VOCAB_SIZE, dec_voc_size=VOCAB_SIZE, d_model=D_MODEL,
        n_head=N_HEAD, max_len=MAX_LEN, ffn_hidden=FFN_HIDDEN,
        n_layers=N_LAYERS, drop_prob=DROP_PROB, device=DEVICE,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Checkpoint Resuming Logic
    current_step = 0
    latest_checkpoint_path = None
    if RANK == 0 and os.path.exists(MODEL_SAVE_DIR):
        checkpoints = [f for f in os.listdir(MODEL_SAVE_DIR) if f.startswith('model_step_') and f.endswith('.pt')]
        if checkpoints:
            latest_step = max([int(f.split('_')[-1].split('.')[0]) for f in checkpoints])
            latest_checkpoint_path = os.path.join(MODEL_SAVE_DIR, CHECKPOINT_FILENAME_TEMPLATE.format(latest_step))
    
    path_list = [latest_checkpoint_path] if RANK == 0 else [None]
    dist.broadcast_object_list(path_list, src=0)
    latest_checkpoint_path = path_list[0]

    if latest_checkpoint_path:
        map_location = {'cuda:0': f'cuda:{local_rank}'}
        checkpoint = torch.load(latest_checkpoint_path, map_location=map_location, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_step = checkpoint['step']
        logger.info(f"[Rank {RANK}] Resumed training from step {current_step}")
    else:
        if RANK == 0: logger.info("No checkpoint found, starting training from scratch.")

    model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    if RANK == 0: logger.info(f"--- Starting Training to {TOTAL_TRAIN_STEPS} steps ---")
    best_bert_f1 = -1.0
    
    train_loop(model, train_dataloader, optimizer, criterion, CLIP, DEVICE, current_step, train_sampler, best_bert_f1, val_dataloader, tokenizer)
    
    if RANK == 0: logger.info("--- Training Finished ---")
    cleanup_ddp()

if __name__ == "__main__":
    # Add a 'generate' method to the model class for inference
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