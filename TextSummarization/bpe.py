# bpe.py
import os
from datasets import load_from_disk
from tokenizers import ByteLevelBPETokenizer

# --- Configuration ---
DATASET_PATH = "cnn_dailymail"
VOCAB_SIZE = 32000
OUTPUT_DIR = "sum-tokenizer"
TEMP_CORPUS_FILE = "sum_corpus.txt"

print("\n--- Step 2: Training Text Summarization Tokenizer ---")

# --- Pre-run Checks ---
if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: Dataset directory './{DATASET_PATH}' not found.")
    print("Please ensure the CNN/DailyMail dataset is available.")
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
print(f"Writing training data to a temporary corpus file: '{TEMP_CORPUS_FILE}'...")
count = 0
with open(TEMP_CORPUS_FILE, "w", encoding="utf-8") as f:
    for example in train_split:
        # Write both source (article) and target (highlights)
        f.write(example['article'] + "\n")
        f.write(example['highlights'] + "\n")
        count += 2
print(f"Wrote {count} lines to the corpus file.")

# 3. Initialize and train the tokenizer
print("Initializing a new Byte-Level BPE tokenizer...")
tokenizer = ByteLevelBPETokenizer()

print(f"Starting tokenizer training with a vocab size of {VOCAB_SIZE}...")
tokenizer.train(
    files=[TEMP_CORPUS_FILE],
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
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
print(f"✅ Success! SUM Tokenizer trained and saved to './{OUTPUT_DIR}'")

# 5. Clean up the temporary corpus file
os.remove(TEMP_CORPUS_FILE)
print(f"Removed temporary corpus file: '{TEMP_CORPUS_FILE}'")