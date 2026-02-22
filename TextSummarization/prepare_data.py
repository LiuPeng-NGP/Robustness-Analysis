# prepare_data.py
import os
import time
from datasets import load_dataset, load_from_disk

# --- Configuration ---
# The data will be saved in a directory with this name in your current workspace.
SAVE_DIRECTORY = "cnn_dailymail"

print(f"This script will download the CNN/DailyMail dataset and save it to './{SAVE_DIRECTORY}'")
print("-" * 50)

# Check if the directory already contains data to avoid re-downloading
if os.path.exists(SAVE_DIRECTORY):
    print(f"Directory '{SAVE_DIRECTORY}' already exists.")
    try:
        # A quick check to see if it's a loadable dataset
        load_from_disk(SAVE_DIRECTORY)
        print("It appears to be a valid dataset directory. Nothing to do.")
        print("If you want to re-download, please delete the 'cnn_dailymail' folder first.")
        exit() # Exit the script
    except Exception:
        print("Directory exists but is not a valid dataset. Proceeding to download.")


# --- Main Download and Save Logic ---
start_time = time.time()

try:
    # 1. Load the dataset. This will download to a central cache first.
    # We use the '3.0.0' version which is standard.
    print("Loading CNN/DailyMail from Hugging Face hub (or cache)...")
    cnn_dailymail_dataset = load_dataset("cnn_dailymail", "3.0.0")
    print("Dataset loaded successfully.")

    # 2. Save the loaded dataset to the specified directory in your workspace.
    print(f"Saving dataset to the './{SAVE_DIRECTORY}' directory...")
    cnn_dailymail_dataset.save_to_disk(SAVE_DIRECTORY)

    end_time = time.time()
    absolute_path = os.path.abspath(SAVE_DIRECTORY)
    
    print("\n" + "=" * 50)
    print(f"✅ Success! The CNN/DailyMail dataset has been saved to:")
    print(f"   {absolute_path}")
    print(f"Time taken: {end_time - start_time:.2f} seconds.")
    print("=" * 50)

    # 3. Inspect and verify the saved data (optional)
    print("\nVerifying the saved dataset by loading it back from disk...")
    local_cnn_dataset = load_from_disk(SAVE_DIRECTORY)
    print("\nCNN/DailyMail Dataset Information (from local copy):")
    print(local_cnn_dataset)

    cnn_train_split = local_cnn_dataset['train']
    print(f"\nNumber of training examples: {len(cnn_train_split)}")
    print("\nExample from training set:")
    print("Article:", cnn_train_split[0]['article'])
    print("\nHighlights:", cnn_train_split[0]['highlights'])


except Exception as e:
    print(f"\nAn error occurred: {e}")