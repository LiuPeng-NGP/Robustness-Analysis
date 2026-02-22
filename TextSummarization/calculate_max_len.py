# calculate_max_len.py
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from tokenizers.implementations import ByteLevelBPETokenizer
from tqdm import tqdm

# --- Configuration for Text Summarization ---
DATASET_PATH = "cnn_dailymail"
VOCAB_FILE = "sum-tokenizer/vocab.json"
MERGES_FILE = "sum-tokenizer/merges.txt"

def calculate_and_visualize_lengths():
    """
    Loads the CNN/DailyMail dataset and its tokenizer, calculates token lengths
    for all splits, and reports statistics and visualizations.
    """
    print("--- Starting Max Length Calculation for CNN/DailyMail Dataset ---")

    # 1. --- Load Dataset and Tokenizer ---
    if not os.path.exists(DATASET_PATH) or not os.path.exists(VOCAB_FILE):
        print(f"Error: Ensure dataset '{DATASET_PATH}' and tokenizer files exist.")
        return

    print(f"Loading dataset from '{DATASET_PATH}'...")
    dataset = load_from_disk(DATASET_PATH)

    print(f"Loading tokenizer from '{VOCAB_FILE}' and '{MERGES_FILE}'...")
    tokenizer = ByteLevelBPETokenizer(
        vocab=VOCAB_FILE,
        merges=MERGES_FILE,
    )

    src_lengths = []
    tgt_lengths = []

    # 2. --- Iterate and Tokenize ---
    for split in dataset.keys():
        print(f"\nProcessing '{split}' split...")
        split_dataset = dataset[split]
        for example in tqdm(split_dataset, desc=f"Tokenizing {split}"):
            # Tokenize source (article)
            src_ids = tokenizer.encode(example['article']).ids
            src_lengths.append(len(src_ids))

            # Tokenize target (highlights/summary)
            tgt_ids = tokenizer.encode(example['highlights']).ids
            tgt_lengths.append(len(tgt_ids))

    # 3. --- Calculate Statistics ---
    print("\n--- Analysis Complete ---")

    # Source (Article) stats
    src_max = np.max(src_lengths)
    src_p95 = int(np.percentile(src_lengths, 95))
    src_p99 = int(np.percentile(src_lengths, 99))
    src_p999 = int(np.percentile(src_lengths, 99.9))

    print("\nSource (Article) Token Lengths:")
    print(f"  - Absolute Max: {src_max}")
    print(f"  - 95th Percentile: {src_p95} (95% of articles are this long or shorter)")
    print(f"  - 99th Percentile: {src_p99}")
    print(f"  - 99.9th Percentile: {src_p999}")

    # Target (Summary) stats
    tgt_max = np.max(tgt_lengths)
    tgt_p95 = int(np.percentile(tgt_lengths, 95))
    tgt_p99 = int(np.percentile(tgt_lengths, 99))
    tgt_p999 = int(np.percentile(tgt_lengths, 99.9))

    print("\nTarget (Summary) Token Lengths:")
    print(f"  - Absolute Max: {tgt_max}")
    print(f"  - 95th Percentile: {tgt_p95}")
    print(f"  - 99th Percentile: {tgt_p99}")
    print(f"  - 99.9th Percentile: {tgt_p999}")

    # 4. --- Interpretation and Recommendation ---
    # NOTE: For summarization, source and target lengths are very different.
    # The `max_len` for the model must accommodate the longer of the two, which is the source.
    suggested_max_len = src_p999 # Base suggestion on the source article length
    
    # Round up to a power of 2 or a clean number
    if suggested_max_len <= 512:
        final_suggestion = 512
    elif suggested_max_len <= 1024:
        final_suggestion = 1024
    else:
        # Handle cases where articles are very long
        final_suggestion = 2048

    print("\n--- Recommendation ---")
    print("For summarization, the source article length dictates the required `max_len`.")
    print(f"Based on the 99.9th percentile of article lengths ({src_p999}), a `max_len` of {final_suggestion} is a robust choice.")
    print("This will require significantly more memory than the translation task.")

    # 5. --- Visualize the Distributions ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Source plot
    ax1.hist(src_lengths, bins=50, color='deepskyblue', alpha=0.7, range=(0, src_p999 + 100)) # Adjust range for better view
    ax1.axvline(src_p99, color='orange', linestyle='--', label=f'99th percentile ({src_p99})')
    ax1.axvline(src_p999, color='red', linestyle='--', label=f'99.9th percentile ({src_p999})')
    ax1.set_title('Source (Article) Length Distribution')
    ax1.set_xlabel('Token Length')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Target plot
    ax2.hist(tgt_lengths, bins=50, color='lightcoral', alpha=0.7)
    ax2.axvline(tgt_p99, color='orange', linestyle='--', label=f'99th percentile ({tgt_p99})')
    ax2.axvline(tgt_p999, color='red', linestyle='--', label=f'99.9th percentile ({tgt_p999})')
    ax2.set_title('Target (Summary) Length Distribution')
    ax2.set_xlabel('Token Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Token Length Distributions for CNN/DailyMail Dataset', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("\nDisplaying length distribution plot...")
    plt.show()


if __name__ == "__main__":
    calculate_and_visualize_lengths()
    
    
'''
--- Analysis Complete ---

Source (Article) Token Lengths:
  - Absolute Max: 5821
  - 95th Percentile: 1711 (95% of articles are this long or shorter)
  - 99th Percentile: 2102
  - 99.9th Percentile: 2343

Target (Summary) Token Lengths:
  - Absolute Max: 2536
  - 95th Percentile: 116
  - 99th Percentile: 153
  - 99.9th Percentile: 225

--- Recommendation ---
For summarization, the source article length dictates the required `max_len`.
Based on the 99.9th percentile of article lengths (2343), a `max_len` of 2048 is a robust choice.
This will require significantly more memory than the translation task.
'''