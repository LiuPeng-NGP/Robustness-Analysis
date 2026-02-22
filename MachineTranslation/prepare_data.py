# prepare.py
import os
import time
from datasets import load_dataset

# --- Configuration ---
# The data will be saved in a directory with this name in your current workspace.
SAVE_DIRECTORY = "wmt14"

print(f"This script will download the WMT'14 dataset and save it to './{SAVE_DIRECTORY}'")
print("-" * 50)

# Check if the directory already contains data to avoid re-downloading
if os.path.exists(SAVE_DIRECTORY):
    print(f"Directory '{SAVE_DIRECTORY}' already exists.")
    # You could add a check here to see if it's a valid dataset directory
    try:
        # A quick check to see if it's a loadable dataset
        load_dataset(SAVE_DIRECTORY)
        print("It appears to be a valid dataset directory. Nothing to do.")
        print("If you want to re-download, please delete the 'wmt14' folder first.")
        exit() # Exit the script
    except Exception:
        print("Directory exists but is not a valid dataset. Proceeding to download.")


# --- Main Download and Save Logic ---
start_time = time.time()

try:
    # 1. Load the dataset. This will download to a central cache first.
    print("Loading WMT'14 from Hugging Face hub (or cache)...")
    wmt14_dataset = load_dataset("wmt14", "de-en")
    print("Dataset loaded successfully.")

    # 2. Save the loaded dataset to the specified directory in your workspace.
    print(f"Saving dataset to the './{SAVE_DIRECTORY}' directory...")
    wmt14_dataset.save_to_disk(SAVE_DIRECTORY)

    end_time = time.time()
    absolute_path = os.path.abspath(SAVE_DIRECTORY)
    
    print("\n" + "=" * 50)
    print(f"✅ Success! The WMT'14 dataset has been saved to:")
    print(f"   {absolute_path}")
    print(f"Time taken: {end_time - start_time:.2f} seconds.")
    print("=" * 50)


except Exception as e:
    print(f"\nAn error occurred: {e}")

