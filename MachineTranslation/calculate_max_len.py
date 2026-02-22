# calculate_max_len.py
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
# CHANGE 1: Import the specific tokenizer class
from tokenizers.implementations import ByteLevelBPETokenizer
from tqdm import tqdm

# --- Configuration ---
DATASET_PATH = "wmt14-subset"
# CHANGE 2: Point to the individual files, we'll use them below
VOCAB_FILE = "mt-tokenizer/vocab.json"
MERGES_FILE = "mt-tokenizer/merges.txt"

def calculate_and_visualize_lengths():
    """
    Loads the dataset and tokenizer, calculates token lengths for all splits,
    and reports statistics and visualizations.
    """
    print("--- Starting Max Length Calculation for WMT Dataset ---")

    # 1. --- Load Dataset and Tokenizer ---
    if not os.path.exists(DATASET_PATH) or not os.path.exists(VOCAB_FILE):
        print(f"Error: Ensure dataset '{DATASET_PATH}' and tokenizer files exist.")
        return

    print(f"Loading dataset from '{DATASET_PATH}'...")
    dataset = load_from_disk(DATASET_PATH)
    
    # CHANGE 3: Use the correct constructor to load the tokenizer
    print(f"Loading tokenizer from '{VOCAB_FILE}' and '{MERGES_FILE}'...")
    tokenizer = ByteLevelBPETokenizer(
        vocab=VOCAB_FILE,
        merges=MERGES_FILE,
    )

    src_lengths = []
    tgt_lengths = []

    # 2. --- Iterate and Tokenize ---
    # The rest of the script remains exactly the same.
    for split in dataset.keys():
        print(f"\nProcessing '{split}' split...")
        split_dataset = dataset[split]
        for example in tqdm(split_dataset, desc=f"Tokenizing {split}"):
            # Tokenize source (English)
            src_ids = tokenizer.encode(example['translation']['en']).ids
            src_lengths.append(len(src_ids))

            # Tokenize target (German)
            tgt_ids = tokenizer.encode(example['translation']['de']).ids
            tgt_lengths.append(len(tgt_ids))

    # 3. --- Calculate Statistics (No changes here) ---
    print("\n--- Analysis Complete ---")
    
    src_max = np.max(src_lengths)
    src_p999 = int(np.percentile(src_lengths, 99.9))
    
    print("\nSource (English) Token Lengths:")
    print(f"  - Absolute Max: {src_max}")
    print(f"  - 99.9th Percentile: {src_p999}")

    tgt_max = np.max(tgt_lengths)
    tgt_p999 = int(np.percentile(tgt_lengths, 99.9))

    print("\nTarget (German) Token Lengths:")
    print(f"  - Absolute Max: {tgt_max}")
    print(f"  - 99.9th Percentile: {tgt_p999}")

    suggested_max_len = max(src_p999, tgt_p999)
    if suggested_max_len <= 256: final_suggestion = 256
    elif suggested_max_len <= 512: final_suggestion = 512
    else: final_suggestion = 1024

    print("\n--- Recommendation ---")
    print(f"Based on the 99.9th percentile, a `max_len` of {final_suggestion} is a robust choice for your model.")

    # 4. --- Visualize (No changes here) ---
    plt.figure(figsize=(12, 5))
    plt.hist(src_lengths, bins=100, alpha=0.7, label=f'Source (English) - Max: {src_max}')
    plt.hist(tgt_lengths, bins=100, alpha=0.7, label=f'Target (German) - Max: {tgt_max}')
    plt.axvline(final_suggestion, color='red', linestyle='--', label=f'Suggested max_len: {final_suggestion}')
    plt.title('Token Length Distribution for WMT Dataset')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    print("\nDisplaying length distribution plot...")
    plt.show()


if __name__ == "__main__":
    calculate_and_visualize_lengths()
    


'''
--- Analysis Complete ---

Source (English) Token Lengths:
  - Absolute Max: 5104
  - 99.9th Percentile: 153

Target (German) Token Lengths:
  - Absolute Max: 8646
  - 99.9th Percentile: 156

--- Recommendation ---
Based on the 99.9th percentile, a `max_len` of 256 is a robust choice for your model.
'''