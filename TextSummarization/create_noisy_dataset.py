# create_noise_dataset.py
# for text summarization

import os
import torch
from datasets import load_from_disk, DatasetDict, concatenate_datasets, Dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tqdm import tqdm
import shutil

# DDP (Distributed Data Parallel) Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import your custom Transformer model definition
from transformer import Transformer

# --- DDP Setup and Cleanup Functions ---
def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

# --- 1. Configuration (Adapted for Text Summarization) ---
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
DEVICE = f"cuda:{LOCAL_RANK}"

# --- MODIFIED: Paths and Naming for Summarization ---
MODEL_NAME = "sum_clean_teacher"
RESULTS_DIR = "results"
CHECKPOINT_STEP = 5000 # Use an early-stage checkpoint as the "noisy teacher"

SOURCE_DATASET_PATH = "cnn_dailymail"
VOCAB_FILE = "sum-tokenizer/vocab.json"
MERGES_FILE = "sum-tokenizer/merges.txt"
MODEL_SAVE_DIR = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_checkpoints")
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, f"model_step_{CHECKPOINT_STEP}.pt")

OUTPUT_DATASET_PATH = f"cnn_dailymail_with_noisy_selector_{CHECKPOINT_STEP}k"
TEMP_SHARD_DIR = f"temp_sum_shards_{CHECKPOINT_STEP}k"

# --- MODIFIED: Hyperparameters for Summarization ---
# NOTE: BATCH_SIZE is much smaller due to the massive increase in MAX_LEN
BATCH_SIZE = 16
MAX_LEN = 2048 # To handle long source articles
VOCAB_SIZE = 32000
D_MODEL = 512
N_LAYERS = 6
N_HEAD = 8
FFN_HIDDEN = 2048
DROP_PROB = 0.1
PAD_IDX = 1
SOS_IDX = 0
EOS_IDX = 2

if RANK == 0:
    print(f"Starting DDP generation for Summarization with {WORLD_SIZE} GPUs.")
    if not os.path.exists(CHECKPOINT_PATH): raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    if not os.path.exists(SOURCE_DATASET_PATH): raise FileNotFoundError(f"Source dataset not found at {SOURCE_DATASET_PATH}")
    if os.path.exists(OUTPUT_DATASET_PATH):
        print(f"✅ Output directory '{OUTPUT_DATASET_PATH}' already exists. Skipping.")
        exit()
    os.makedirs(TEMP_SHARD_DIR, exist_ok=True)

def main():
    setup_ddp()

    if RANK == 0: print("\n--- Loading tokenizer and model on all processes ---")
    tokenizer = ByteLevelBPETokenizer(vocab=VOCAB_FILE, merges=MERGES_FILE)
    # No padding here; it will be handled dynamically in the batch.
    tokenizer.enable_truncation(max_length=MAX_LEN)
    
    model = Transformer(
        src_pad_idx=PAD_IDX, trg_pad_idx=PAD_IDX, trg_sos_idx=SOS_IDX,
        enc_voc_size=VOCAB_SIZE, dec_voc_size=VOCAB_SIZE, d_model=D_MODEL,
        n_head=N_HEAD, max_len=MAX_LEN, ffn_hidden=FFN_HIDDEN,
        n_layers=N_LAYERS, drop_prob=DROP_PROB, device=DEVICE,
    ).to(DEVICE)
    
    # Attach the same generate method
    def generate(self, src, max_len, sos_idx, eos_idx):
        self.eval(); batch_size = src.shape[0]
        trg = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=self.device)
        src_mask = self.make_src_mask(src); enc_src = self.encoder(src, src_mask)
        for _ in range(max_len - 1):
            trg_mask = self.make_trg_mask(trg)
            output = self.decoder(trg, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(2)[:, -1].unsqueeze(1)
            trg = torch.cat((trg, pred_token), dim=1)
            # Stop if all sequences in the batch have generated the EOS token
            if (trg == eos_idx).any(dim=-1).sum() == batch_size: break
        return trg
    Transformer.generate = generate

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = DDP(model, device_ids=[LOCAL_RANK])
    model.eval()
    if RANK == 0: print("Model loaded and wrapped in DDP successfully.")

    if RANK == 0: print(f"\n--- Loading source dataset from '{SOURCE_DATASET_PATH}' ---")
    source_dataset = load_from_disk(SOURCE_DATASET_PATH)

    # --- MODIFIED: Unified processing function for Summarization ---
    def process_batch(batch, indices, split_name):
        """Generates noisy summaries, adds selectors, and restructures the batch."""
        # 1. Generate noisy targets from the 'article' column
        sources = batch['article']
        
        # Tokenize sources WITHOUT padding first to handle dynamic batching
        inputs = tokenizer.encode_batch(sources)
        # Manually pad and convert to tensor
        max_batch_len = max(len(encoding.ids) for encoding in inputs)
        padded_ids = [encoding.ids + [PAD_IDX] * (max_batch_len - len(encoding.ids)) for encoding in inputs]
        input_ids = torch.tensor(padded_ids, dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            # IMPORTANT: Set a reasonable max_len for the *output summary*
            # This prevents run-on generation and speeds up the process significantly.
            outputs = model.module.generate(input_ids, max_len=256, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
        
        noisy_summaries = tokenizer.decode_batch(outputs.cpu().numpy(), skip_special_tokens=True)
        
        # 2. Restructure the output to the new format
        new_batch = {
            "article": batch["article"],
            "highlights": batch["highlights"],
            "highlights_noisy": noisy_summaries
        }
        if split_name == 'train':
            new_batch["noise_selector"] = [idx % 100 for idx in indices]
            
        return new_batch

    if RANK == 0: print(f"\n--- Generating noisy summaries in parallel across {WORLD_SIZE} GPUs ---")
    
    for split in source_dataset.keys():
        split_shard = source_dataset[split].shard(num_shards=WORLD_SIZE, index=RANK)
        shard_save_path = os.path.join(TEMP_SHARD_DIR, f"{split}_shard_{RANK}")
        
        print(f"[GPU {RANK}] Processing {len(split_shard)} examples for '{split}' split...")

        # --- MODIFIED: .map() call for Summarization ---
        processed_shard = split_shard.map(
            process_batch,
            with_indices=True,
            batched=True,
            batch_size=BATCH_SIZE,
            remove_columns=source_dataset[split].column_names, # Remove old columns ('article', 'highlights', 'id')
            fn_kwargs={"split_name": split}
        )

        processed_shard.save_to_disk(shard_save_path)
        print(f"[GPU {RANK}] Finished and saved shard to '{shard_save_path}'")

    dist.barrier()

    if RANK == 0:
        print("\n--- All GPUs finished. Aggregating results on Rank 0 ---")
        final_dataset_dict = DatasetDict()

        # Determine the splits to process from the source dataset
        splits_to_process = source_dataset.keys()

        for split in splits_to_process:
            print(f"\n--- Aggregating '{split}' split ---")
            shard_list = []
            found_shards = 0
            
            # Loop through all possible shard paths
            for i in range(WORLD_SIZE):
                shard_path = os.path.join(TEMP_SHARD_DIR, f"{split}_shard_{i}")
                
                # The crucial fix: Check if the shard directory actually exists
                if os.path.exists(shard_path):
                    print(f"  [OK] Found and loading shard: {shard_path}")
                    shard_list.append(Dataset.load_from_disk(shard_path))
                    found_shards += 1
                else:
                    print(f"  [WARN] Shard not found, skipping: {shard_path}")

            # Only proceed if we found at least one valid shard for this split
            if shard_list:
                print(f"  > Found {found_shards}/{WORLD_SIZE} shards for '{split}'. Concatenating...")
                final_dataset_dict[split] = concatenate_datasets(shard_list)
            else:
                print(f"  > No shards were found for '{split}'. This split will be excluded from the final dataset.")

        # Final check before saving
        if not final_dataset_dict:
            print("\nError: No valid data was aggregated. Halting before saving.")
            # Optional: Decide if you want to clean up or inspect TEMP_SHARD_DIR
            # shutil.rmtree(TEMP_SHARD_DIR) 
            return # Exit the function

        print(f"\n--- Saving final aggregated dataset to '{OUTPUT_DATASET_PATH}' ---")
        final_dataset_dict.save_to_disk(OUTPUT_DATASET_PATH)

        print(f"\n--- Cleaning up temporary shard directories: '{TEMP_SHARD_DIR}' ---")
        shutil.rmtree(TEMP_SHARD_DIR)

        print("\n" + "="*50)
        print("✅ Success! DDP-generated dataset for summarization created.")
        print(f"   Saved to: {os.path.abspath(OUTPUT_DATASET_PATH)}")
        print("="*50)

        print("\nVerifying the saved dataset by loading it back from disk...")
        reloaded_dataset = load_from_disk(OUTPUT_DATASET_PATH)
        print("\nNew Dataset Information:")
        print(reloaded_dataset)
        
        if 'train' in reloaded_dataset:
            print("\nExample from 'train' split:")
            example = reloaded_dataset['train'][105]
            print({k: v for k, v in example.items()}) # Print entire example dict
        
        if 'validation' in reloaded_dataset:
            print("\nExample from 'validation' split (should not have selector):")
            val_example = reloaded_dataset['validation'][0]
            print({k: v for k, v in val_example.items()})


    cleanup_ddp()

if __name__ == "__main__":
    main()