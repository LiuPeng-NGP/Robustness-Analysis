# get_subset.py
import os
import time
from datasets import load_from_disk, DatasetDict

# --- Configuration ---
SOURCE_DIRECTORY = "wmt14"
SAVE_DIRECTORY = "wmt14-subset"
SUBSET_SIZE = 287113 # Matching the CNN/DailyMail training set size

print(f"This script will create a subset of the WMT'14 dataset.")
print(f"Source: './{SOURCE_DIRECTORY}'")
print(f"Destination: './{SAVE_DIRECTORY}'")
print(f"Training examples to select: {SUBSET_SIZE}")
print("-" * 50)

# --- Pre-run Checks ---

# 1. Check if the destination directory already exists
if os.path.exists(SAVE_DIRECTORY):
    print(f"Directory '{SAVE_DIRECTORY}' already exists.")
    try:
        # A quick check to see if it's a loadable dataset
        load_from_disk(SAVE_DIRECTORY)
        print("It appears to be a valid dataset directory. Nothing to do.")
        print(f"If you want to re-create the subset, please delete the '{SAVE_DIRECTORY}' folder first.")
        exit()
    except Exception:
        print("Directory exists but is not a valid dataset. Proceeding to create subset.")

# 2. Check if the source directory exists
if not os.path.exists(SOURCE_DIRECTORY):
    print(f"❌ Error: Source directory './{SOURCE_DIRECTORY}' not found.")
    print("Please make sure you have already downloaded the full WMT'14 dataset using the previous script.")
    exit()


# --- Main Logic ---
start_time = time.time()

try:
    # 1. Load the full WMT'14 dataset from the local disk
    print(f"Loading full WMT'14 dataset from './{SOURCE_DIRECTORY}'...")
    full_wmt14_dataset = load_from_disk(SOURCE_DIRECTORY)
    print("Full dataset loaded successfully.")
    print("\nOriginal Dataset Information:")
    print(full_wmt14_dataset)

    # 2. Get the training split and verify its size
    wmt_train_full = full_wmt14_dataset['train']
    if len(wmt_train_full) < SUBSET_SIZE:
        print(f"❌ Error: The original training set has only {len(wmt_train_full)} examples, which is less than the requested subset size of {SUBSET_SIZE}.")
        exit()

    # 3. Create the random subset of the training data
    # We shuffle with a fixed seed for reproducibility
    print(f"\nCreating a random subset of {SUBSET_SIZE} training examples...")
    wmt_train_subset = wmt_train_full.shuffle(seed=42).select(range(SUBSET_SIZE))
    print("Subset created.")

    # 4. Combine the new training subset with the original validation and test sets
    wmt14_subset = DatasetDict({
        'train': wmt_train_subset,
        'validation': full_wmt14_dataset['validation'],
        'test': full_wmt14_dataset['test']
    })

    # 5. Save the new subset dataset to disk
    print(f"Saving subset dataset to './{SAVE_DIRECTORY}'...")
    wmt14_subset.save_to_disk(SAVE_DIRECTORY)

    end_time = time.time()
    absolute_path = os.path.abspath(SAVE_DIRECTORY)

    print("\n" + "=" * 50)
    print(f"✅ Success! The WMT'14 subset has been saved to:")
    print(f"   {absolute_path}")
    print(f"Time taken: {end_time - start_time:.2f} seconds.")
    print("=" * 50)

    # 6. Verify the saved data
    print("\nVerifying the saved subset by loading it back from disk...")
    local_wmt_subset = load_from_disk(SAVE_DIRECTORY)
    print("\nWMT'14 Subset Dataset Information (from local copy):")
    print(local_wmt_subset)

    print(f"\nNumber of new training examples: {len(local_wmt_subset['train'])}")
    print("\nExample from the new training set:")
    print(local_wmt_subset['train'][0]['translation'])


except Exception as e:
    print(f"\nAn error occurred: {e}")