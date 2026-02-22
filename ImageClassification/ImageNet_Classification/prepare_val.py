# prepare_val.py
import os
import scipy.io
import shutil

# Define paths relative to the script's location or use absolute paths
imagenet_data_dir = './data' # Assumes script is in the 'data' directory
devkit_path = os.path.join(imagenet_data_dir, 'ILSVRC2012_devkit_t12')
val_dir = os.path.join(imagenet_data_dir, 'val')
meta_path = os.path.join(devkit_path, 'data', 'meta.mat')
ground_truth_path = os.path.join(devkit_path, 'data', 'ILSVRC2012_validation_ground_truth.txt')

# Check if directories/files exist
if not os.path.isdir(val_dir):
    print(f"ERROR: Validation directory not found at {val_dir}")
    exit(1)
if not os.path.isfile(meta_path):
     print(f"ERROR: meta.mat not found at {meta_path}")
     print("Ensure ILSVRC2012_devkit_t12.tar.gz was extracted correctly.")
     exit(1)
if not os.path.isfile(ground_truth_path):
    print(f"ERROR: Validation ground truth file not found at {ground_truth_path}")
    print("Ensure ILSVRC2012_devkit_t12.tar.gz was extracted correctly.")
    exit(1)


print("Loading metadata...")
# Load meta.mat to get synset names
meta = scipy.io.loadmat(meta_path)
# Structure is a bit nested, adjust based on your meta.mat inspection if needed
synsets = meta['synsets']

# Create a mapping from ID to synset name (folder name)
id_to_synset = {}
for i in range(len(synsets)):
     # Assuming ILSVRC2012_ID is the correct field, might vary slightly
     # Check the structure of your meta['synsets'] if this fails
    if len(synsets[i]) > 0 and len(synsets[i][0]) > 0:
         ilsvrc_id = synsets[i][0][0][0][0] # Access ILSVRC2012_ID
         synset_name = synsets[i][0][1][0]  # Access WNID (like n0******)
         id_to_synset[ilsvrc_id] = synset_name
    else:
         print(f"Warning: Unexpected structure for synset index {i}")


print("Loading ground truth labels...")
# Load ground truth labels
with open(ground_truth_path, 'r') as f:
    ground_truth_ids = [int(line.strip()) for line in f.readlines()]

print(f"Found {len(ground_truth_ids)} validation labels.")
if len(ground_truth_ids) != 50000:
    print(f"Warning: Expected 50000 validation labels, found {len(ground_truth_ids)}")

print("Processing validation images...")
# Get list of validation images (assuming they are sorted correctly)
val_images = sorted([f for f in os.listdir(val_dir) if f.endswith('.JPEG')])

if len(val_images) != len(ground_truth_ids):
     print(f"ERROR: Number of validation images ({len(val_images)}) does not match number of labels ({len(ground_truth_ids)})!")
     exit(1)

# Create subdirectories and move files
images_moved = 0
for i, img_filename in enumerate(val_images):
    label_id = ground_truth_ids[i]
    if label_id not in id_to_synset:
         print(f"Warning: Label ID {label_id} not found in meta.mat mapping for image {img_filename}. Skipping.")
         continue

    synset_folder_name = id_to_synset[label_id]
    target_folder = os.path.join(val_dir, synset_folder_name)

    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Construct source and destination paths
    source_path = os.path.join(val_dir, img_filename)
    dest_path = os.path.join(target_folder, img_filename)

    # Move the file
    try:
        if os.path.isfile(source_path): # Make sure it's a file before moving
             shutil.move(source_path, dest_path)
             images_moved += 1
        # else: It might have already been moved if script is re-run partially
        #     print(f"Warning: Source file {source_path} not found (already moved?).")

    except Exception as e:
        print(f"Error moving {source_path} to {dest_path}: {e}")

    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{len(val_images)} images...")

print(f"Finished. Moved {images_moved} validation images into class subfolders within {val_dir}.")
# Optional: Check if any .JPEG files are left directly in val_dir
remaining_files = [f for f in os.listdir(val_dir) if os.path.isfile(os.path.join(val_dir, f)) and f.endswith('.JPEG')]
if remaining_files:
    print(f"Warning: {len(remaining_files)} JPEG files remain directly in {val_dir}. Check for errors.")
    # print(remaining_files[:10]) # Print a few examples