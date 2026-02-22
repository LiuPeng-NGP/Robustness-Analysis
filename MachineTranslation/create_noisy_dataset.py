# create_noisy_dataset.py
# for machine translation
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

# --- 1. Configuration ---
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
DEVICE = f"cuda:{LOCAL_RANK}"

MODEL_NAME = "mt_clean_teacher"
RESULTS_DIR = "results"
CHECKPOINT_STEP = 5000

SOURCE_DATASET_PATH = "wmt14-subset"
VOCAB_FILE = "mt-tokenizer/vocab.json"
MERGES_FILE = "mt-tokenizer/merges.txt"
MODEL_SAVE_DIR = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_checkpoints")
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, f"model_step_{CHECKPOINT_STEP}.pt")

OUTPUT_DATASET_PATH = f"wmt14_with_noisy_selector_{CHECKPOINT_STEP}k"
TEMP_SHARD_DIR = f"temp_shards_{CHECKPOINT_STEP}k" # Directory for final shard folders

BATCH_SIZE = 64
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

if RANK == 0:
    print(f"Starting DDP generation with {WORLD_SIZE} GPUs.")
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
    tokenizer.enable_padding(pad_id=PAD_IDX, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=MAX_LEN)
    
    model = Transformer(
        src_pad_idx=PAD_IDX, trg_pad_idx=PAD_IDX, trg_sos_idx=SOS_IDX,
        enc_voc_size=VOCAB_SIZE, dec_voc_size=VOCAB_SIZE, d_model=D_MODEL,
        n_head=N_HEAD, max_len=MAX_LEN, ffn_hidden=FFN_HIDDEN,
        n_layers=N_LAYERS, drop_prob=DROP_PROB, device=DEVICE,
    ).to(DEVICE)

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

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = DDP(model, device_ids=[LOCAL_RANK])
    model.eval()
    if RANK == 0: print("Model loaded and wrapped in DDP successfully.")

    if RANK == 0: print(f"\n--- Loading source dataset from '{SOURCE_DATASET_PATH}' ---")
    source_dataset = load_from_disk(SOURCE_DATASET_PATH)

    # *** FIX: Create a single, unified processing function ***
    def process_batch(batch, indices, split_name):
        """Generates noise, adds selectors, and restructures the batch all at once."""
        # 1. Generate noisy targets
        sources = [item['en'] for item in batch['translation']]
        inputs = tokenizer.encode_batch(sources)
        input_ids = torch.tensor([encoding.ids for encoding in inputs], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            outputs = model.module.generate(input_ids, max_len=MAX_LEN, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
        noisy_targets = tokenizer.decode_batch(outputs.cpu().numpy(), skip_special_tokens=True)
        
        # 2. Restructure the output
        new_translations = []
        for i in range(len(sources)):
            item = batch['translation'][i]
            restructured_item = {
                "en": item["en"],
                "de": item["de"],
                "de_noisy": noisy_targets[i]
            }
            if split_name == 'train':
                restructured_item["noise_selector"] = indices[i] % 100
            new_translations.append(restructured_item)
            
        return {"translation": new_translations}

    if RANK == 0: print(f"\n--- Generating noisy targets in parallel across {WORLD_SIZE} GPUs ---")
    
    for split in source_dataset.keys():
        split_shard = source_dataset[split].shard(num_shards=WORLD_SIZE, index=RANK)
        
        # *** FIX: Define a clean DIRECTORY path for saving the final shard ***
        shard_save_path = os.path.join(TEMP_SHARD_DIR, f"{split}_shard_{RANK}")
        
        print(f"[GPU {RANK}] Processing {len(split_shard)} examples for '{split}' split...")

        # *** FIX: Use a single .map() call with the unified function ***
        processed_shard = split_shard.map(
            process_batch,
            with_indices=True,
            batched=True,
            batch_size=BATCH_SIZE,
            remove_columns=source_dataset[split].column_names, # Remove old columns
            fn_kwargs={"split_name": split} # Pass split name to the function
        )

        # Now save the final, processed shard to its own directory
        processed_shard.save_to_disk(shard_save_path)
        print(f"[GPU {RANK}] Finished and saved shard to '{shard_save_path}'")

    dist.barrier()

    if RANK == 0:
        print("\n--- All GPUs finished. Aggregating results on Rank 0 ---")
        final_dataset_dict = DatasetDict()

        for split in source_dataset.keys():
            shard_list = []
            for i in range(WORLD_SIZE):
                shard_path = os.path.join(TEMP_SHARD_DIR, f"{split}_shard_{i}")
                shard_list.append(Dataset.load_from_disk(shard_path))
            
            final_dataset_dict[split] = concatenate_datasets(shard_list)

        print(f"\n--- Saving final aggregated dataset to '{OUTPUT_DATASET_PATH}' ---")
        final_dataset_dict.save_to_disk(OUTPUT_DATASET_PATH)

        print("\n--- Cleaning up temporary shard directories ---")
        shutil.rmtree(TEMP_SHARD_DIR)

        print("\n" + "="*50)
        print("✅ Success! DDP-generated dataset with noise selector created.")
        print(f"   Saved to: {os.path.abspath(OUTPUT_DATASET_PATH)}")
        print("="*50)

        reloaded_dataset = load_from_disk(OUTPUT_DATASET_PATH)
        print("\nNew Dataset Information:")
        print(reloaded_dataset)
        print("\nExample from 'train' split:")
        print(reloaded_dataset['train'][105])
        print("\nExample from 'validation' split:")
        print(reloaded_dataset['validation'][0])

    cleanup_ddp()

if __name__ == "__main__":
    main()