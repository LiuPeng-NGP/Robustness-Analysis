# train_low_quality.py
# for machine translation
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
from itertools import islice  # <-- FIX: Added this import

# DDP (Distributed Data Parallel) Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import datetime

# --- Metric Imports ---
import sacrebleu
from bert_score import score as bert_score_calc

# Import your custom Transformer model
from transformer import Transformer

# --- DDP Setup and Cleanup Functions ---
def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

# --- 1. Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer model with a specific noise ratio and step count.")
    parser.add_argument(
        "--noise_ratio",
        type=float,
        required=True,
        help="Fraction of noisy data to use for training (a float from 0.0 to 1.0)."
    )
    # MODIFIED: Changed from base_steps to a direct total_steps parameter
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
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
CLIP = 1.0
EVAL_STEPS = 5000
NUM_EVAL_SAMPLES = 256

# Model Hyperparameters
VOCAB_SIZE = 32000
MAX_LEN = 256
D_MODEL = 512
N_LAYERS = 6
N_HEAD = 8
FFN_HIDDEN = 2048
DROP_PROB = 0.1
PAD_IDX = 1
SOS_IDX = 0
EOS_IDX = 2

# --- 3. Data Loading and Collation (No changes) ---
def collate_fn(batch, tokenizer, noise_ratio, is_train):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(item['translation']['en'])
        if is_train:
            selector = item['translation']['noise_selector']
            if selector < (noise_ratio * 100):
                tgt_batch.append(item['translation']['de_noisy'])
            else:
                tgt_batch.append(item['translation']['de'])
        else:
            tgt_batch.append(item['translation']['de'])
    src_tokenized = tokenizer.encode_batch(src_batch)
    tgt_tokenized = tokenizer.encode_batch(tgt_batch)
    src_tensor = torch.tensor([encoding.ids for encoding in src_tokenized], dtype=torch.long)
    tgt_tensor = torch.tensor([encoding.ids for encoding in tgt_tokenized], dtype=torch.long)
    return src_tensor, tgt_tensor

def get_data_loader(tokenizer, batch_size, noise_ratio, dataset_path):
    if not os.path.exists(dataset_path):
        if RANK == 0: logging.getLogger(__name__).error(f"Dataset not found at {dataset_path}.")
        exit()
    if RANK == 0: logging.getLogger(__name__).info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    train_collate = partial(collate_fn, tokenizer=tokenizer, noise_ratio=noise_ratio, is_train=True)
    val_collate = partial(collate_fn, tokenizer=tokenizer, noise_ratio=0, is_train=False)
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
def run_evaluation(model, val_dataloader, criterion, tokenizer, device, current_step, best_nll, results_csv_file, best_model_path, logger):
    model.eval()
    nll_loss = calculate_nll(model, val_dataloader, criterion, device)
    bleu, bert_f1 = generate_and_score(model, val_dataloader, tokenizer, device)
    logger.info(f"\tValidation NLL Loss: {nll_loss:.4f} (Best: {best_nll:.4f})")
    logger.info(f"\tValidation SacreBLEU: {bleu:.2f}")
    logger.info(f"\tValidation BERTScore F1: {bert_f1:.4f}")

    if RANK == 0:
        with open(results_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_step, bleu, bert_f1, nll_loss])
        if nll_loss < best_nll:
            best_nll = nll_loss
            torch.save(model.module.state_dict(), best_model_path)
            logger.info(f"🎉 New best model saved with NLL: {nll_loss:.4f} at step {current_step}")

    model.train()
    return best_nll

def train_loop(model, dataloader, optimizer, criterion, device, current_step, train_sampler, best_nll, val_dataloader, tokenizer, paths, logger, total_steps):
    model.train()
    epoch = 1
    
    while current_step < total_steps:
        if RANK == 0: logger.info(f"--- Starting Epoch {epoch} | Current Step: {current_step}/{total_steps} ---")
        train_sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=(RANK != 0))

        for src, trg in pbar:
            if current_step >= total_steps: break
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
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
                if current_step % EVAL_STEPS == 0:
                    logger.info(f"\n--- Evaluation at step {current_step} ---")
                    best_nll = run_evaluation(
                        model, val_dataloader, criterion, tokenizer, device,
                        current_step, best_nll, paths['csv'], paths['best_model'], logger
                    )
        dist.barrier()
        epoch += 1
    return current_step

@torch.no_grad()
def calculate_nll(model, dataloader, criterion, device):
    model.eval()
    total_loss, num_batches = 0, 0
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

@torch.no_grad()
def generate_and_score(model, dataloader, tokenizer, device):
    model.eval()
    candidate_strings, reference_strings = [], []
    num_batches_to_eval = (NUM_EVAL_SAMPLES // BATCH_SIZE) // dist.get_world_size()
    val_subset = list(islice(dataloader, num_batches_to_eval))
    for src, trg in val_subset:
        src = src.to(device)
        for ref_ids in trg.cpu().numpy():
            reference_strings.append(tokenizer.decode(ref_ids, skip_special_tokens=True))
        batch_translations = model.module.generate(src, max_len=MAX_LEN, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
        for trans_ids in batch_translations.cpu().numpy():
            candidate_strings.append(tokenizer.decode(trans_ids, skip_special_tokens=True))
    if not candidate_strings or not reference_strings: return 0.0, 0.0
    bleu = sacrebleu.corpus_bleu(candidate_strings, [reference_strings])
    _, _, F1 = bert_score_calc(candidate_strings, reference_strings, lang="de", verbose=(RANK==0))
    return bleu.score, F1.mean().item()

# --- 5. Main Execution ---
def main():
    args = parse_args()
    setup_ddp()
    DEVICE = f"cuda:{os.environ['LOCAL_RANK']}"

    MODEL_NAME = f"mt_noise_{args.noise_ratio}"
    RESULTS_DIR = "results"
    MODEL_SAVE_DIR = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_checkpoints")
    
    paths = {
        'log': os.path.join(RESULTS_DIR, f"{MODEL_NAME}_training.log"),
        'csv': os.path.join(RESULTS_DIR, f"{MODEL_NAME}_scores.csv"),
        'best_model': os.path.join(MODEL_SAVE_DIR, "best_model.pt"),
        'dataset': f"wmt14_with_noisy_selector_5000k"
    }

    if RANK == 0: os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    logger = logging.getLogger(f"NoiseRatio_{args.noise_ratio}")
    if RANK == 0:
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(paths['log'], mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    else:
        logger.setLevel(logging.WARNING)

    logger.info(f"[Rank {RANK}] Using device: {DEVICE}")
    logger.info(f"--- Starting training with Noise Ratio: {args.noise_ratio} ({args.noise_ratio*100:.1f}%) ---")
    logger.info(f"Total training steps for this run: {args.total_steps}")


    if RANK == 0:
        logger.info("--- Pre-caching models on main process (rank 0) ---")
        bert_score_calc(["."], [","], lang="de", verbose=False)
        logger.info("BERTScore model cached.")
        if not os.path.exists(paths['csv']):
            with open(paths['csv'], 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'sacrebleu', 'bertscore_f1', 'nll_loss'])
    dist.barrier()

    tokenizer = ByteLevelBPETokenizer(vocab="mt-tokenizer/vocab.json", merges="mt-tokenizer/merges.txt")
    tokenizer.enable_padding(pad_id=PAD_IDX, pad_token="<pad>", length=MAX_LEN)
    tokenizer.enable_truncation(max_length=MAX_LEN)
    
    train_dl, val_dl, train_sampler = get_data_loader(tokenizer, BATCH_SIZE, args.noise_ratio, paths['dataset'])
    if RANK == 0: logger.info("DataLoaders created.")

    model = Transformer(
        src_pad_idx=PAD_IDX, trg_pad_idx=PAD_IDX, trg_sos_idx=SOS_IDX,
        enc_voc_size=VOCAB_SIZE, dec_voc_size=VOCAB_SIZE, d_model=D_MODEL,
        n_head=N_HEAD, max_len=MAX_LEN, ffn_hidden=FFN_HIDDEN,
        n_layers=N_LAYERS, drop_prob=DROP_PROB, device=DEVICE,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    if RANK == 0: logger.info(f"--- Starting fresh training up to {args.total_steps} steps ---")
    best_nll = float('inf')
    
    train_loop(
        model, train_dl, optimizer, criterion, DEVICE, 0,
        train_sampler, best_nll, val_dl, tokenizer, paths, logger,
        total_steps=args.total_steps
    )
    
    if RANK == 0: logger.info("--- Training Finished ---")
    cleanup_ddp()

if __name__ == "__main__":
    def generate(self, src, max_len, sos_idx, eos_idx):
        self.eval(); batch_size = src.shape[0]
        trg = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=self.device)
        src_mask = self.make_src_mask(src); enc_src = self.encoder(src, src_mask)
        for _ in range(max_len - 1):
            trg_mask = self.make_trg_mask(trg)
            output = self.decoder(trg, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(2)[:, -1].unsqueeze(1)
            trg = torch.cat((trg, pred_token), dim=1)
            if (trg == eos_idx).any(dim=-1).sum() == batch_size: break
        return trg
    Transformer.generate = generate
    
    main()