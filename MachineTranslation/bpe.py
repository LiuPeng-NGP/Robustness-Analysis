# bpe.py
import os
from datasets import load_from_disk
from tokenizers import ByteLevelBPETokenizer

# --- Configuration ---
DATASET_PATH = "wmt14-subset"
VOCAB_SIZE = 32000
OUTPUT_DIR = "mt-tokenizer"

TEMP_CORPUS_FILE = "mt_corpus.txt"
# The tokenizer trainer reads data from raw text files. This variable defines the
# filename for a temporary file where we will write all sentences from our dataset.
# This file will be created before training and deleted afterward.


print("--- Step 1: Training Machine Translation Tokenizer ---")

# --- Pre-run Checks ---
if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: Dataset directory './{DATASET_PATH}' not found.")
    print("Please ensure the WMT'14 subset has been created successfully.")
    exit()

if os.path.exists(OUTPUT_DIR):
    print(f"✅ Tokenizer directory './{OUTPUT_DIR}' already exists. Skipping training.")
    exit()

# --- Main Logic ---

# 1. Load the training split of the dataset
print(f"Loading dataset from '{DATASET_PATH}'...")
dataset = load_from_disk(DATASET_PATH)
train_split = dataset['train']
print(f"Loaded {len(train_split)} training examples.")

# 2. Prepare the text corpus for the tokenizer
# The tokenizer is trained on raw text. We'll iterate through the dataset
# and write both the English and German sentences to a single file.
print(f"Writing training data to a temporary corpus file: '{TEMP_CORPUS_FILE}'...")
count = 0
with open(TEMP_CORPUS_FILE, "w", encoding="utf-8") as f:
    for example in train_split:
        # Write both source (English) and target (German) sentences
        f.write(example['translation']['en'] + "\n")
        f.write(example['translation']['de'] + "\n")
        count += 2
print(f"Wrote {count} lines to the corpus file.")

# 3. Initialize and train the tokenizer
print("Initializing a new Byte-Level BPE tokenizer...")
# We use ByteLevelBPE for robustness with multiple languages and special characters.
tokenizer = ByteLevelBPETokenizer()

print(f"Starting tokenizer training with a vocab size of {VOCAB_SIZE}...")
tokenizer.train(
    files=[TEMP_CORPUS_FILE],
    vocab_size=VOCAB_SIZE,
    min_frequency=2,  # A token must appear at least twice
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

# 4. Save the tokenizer
os.makedirs(OUTPUT_DIR, exist_ok=True)
tokenizer.save_model(OUTPUT_DIR)
print(f"✅ Success! MT Tokenizer trained and saved to './{OUTPUT_DIR}'")

# 5. Clean up the temporary corpus file
os.remove(TEMP_CORPUS_FILE)
print(f"Removed temporary corpus file: '{TEMP_CORPUS_FILE}'")