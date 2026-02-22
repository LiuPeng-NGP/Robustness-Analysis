# aggregate_shards.py
import os
import shutil
from datasets import DatasetDict, concatenate_datasets, Dataset, load_from_disk

# --- Configuration ---
# Make sure these paths match the ones in your original script
WORLD_SIZE = 8 # IMPORTANT: Set this to the number of GPUs you were using (e.g., 8)
CHECKPOINT_STEP = 5000
TEMP_SHARD_DIR = f"temp_sum_shards_{CHECKPOINT_STEP}k"
OUTPUT_DATASET_PATH = f"cnn_dailymail_with_noisy_selector_{CHECKPOINT_STEP}k"
SOURCE_DATASET_PATH = "cnn_dailymail" # Needed to get the list of splits

def safe_aggregate():
    """
    Aggregates dataset shards from a temporary directory,
    skipping any missing shards.
    """
    if not os.path.exists(TEMP_SHARD_DIR):
        print(f"Error: Temporary shard directory not found at '{TEMP_SHARD_DIR}'")
        return

    print("--- Starting Safe Aggregation Process ---")
    
    # Load the original dataset structure to know which splits to look for
    try:
        source_dataset = load_from_disk(SOURCE_DATASET_PATH)
        splits_to_process = source_dataset.keys()
        print(f"Found splits to process: {list(splits_to_process)}")
    except FileNotFoundError:
        print(f"Original dataset not found at {SOURCE_DATASET_PATH}. Assuming ['train', 'validation', 'test'].")
        splits_to_process = ['train', 'validation', 'test']

    final_dataset_dict = DatasetDict()

    for split in splits_to_process:
        print(f"\n--- Aggregating '{split}' split ---")
        shard_list = []
        found_shards = 0
        for i in range(WORLD_SIZE):
            shard_path = os.path.join(TEMP_SHARD_DIR, f"{split}_shard_{i}")
            
            # This is the crucial fix: check if the shard exists before loading
            if os.path.exists(shard_path):
                print(f"  Loading shard: {shard_path}")
                shard_list.append(Dataset.load_from_disk(shard_path))
                found_shards += 1
            else:
                print(f"  WARNING: Shard not found, skipping: {shard_path}")

        if not shard_list:
            print(f"No shards found for split '{split}'. It will be excluded from the final dataset.")
            continue

        print(f"Found {found_shards}/{WORLD_SIZE} shards for '{split}'. Concatenating...")
        final_dataset_dict[split] = concatenate_datasets(shard_list)

    if not final_dataset_dict:
        print("\nNo data was aggregated. Halting.")
        return

    print(f"\n--- Saving final aggregated dataset to '{OUTPUT_DATASET_PATH}' ---")
    final_dataset_dict.save_to_disk(OUTPUT_DATASET_PATH)

    print(f"\n--- Cleaning up temporary shard directory: '{TEMP_SHARD_DIR}' ---")
    shutil.rmtree(TEMP_SHARD_DIR)

    print("\n" + "="*50)
    print("✅ Success! Your dataset has been recovered and aggregated.")
    print(f"   Saved to: {os.path.abspath(OUTPUT_DATASET_PATH)}")
    print("="*50)

    # Verification
    print("\nVerifying final dataset:")
    reloaded_dataset = load_from_disk(OUTPUT_DATASET_PATH)
    print(reloaded_dataset)
    if 'train' in reloaded_dataset:
        print("\nExample from 'train' split:")
        print(reloaded_dataset['train'][0])

if __name__ == "__main__":
    safe_aggregate()