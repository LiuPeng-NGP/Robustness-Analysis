#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
from pathlib import Path

def get_imagenet_classes(source_train_dir):
    """Gets a sorted list of class WNIDs from the ImageNet train directory."""
    if not os.path.isdir(source_train_dir):
        print(f"ERROR: Source training directory not found: {source_train_dir}")
        return None
    
    try:
        # List directories (which are the class WNIDs)
        classes = sorted([d for d in os.listdir(source_train_dir) if os.path.isdir(os.path.join(source_train_dir, d))])
        if not classes:
            print(f"ERROR: No class subdirectories found in {source_train_dir}")
            return None
        # Basic check for WNID format (optional but helpful)
        if not all(c.startswith('n') and c[1:].isdigit() for c in classes[:10]):
             print(f"Warning: Subdirectories in {source_train_dir} do not look like ImageNet WNIDs (e.g., nxxxxxxxxx).")
        print(f"Found {len(classes)} classes in {source_train_dir}.")
        return classes
    except OSError as e:
        print(f"ERROR: Could not read directory {source_train_dir}: {e}")
        return None

def create_subset_links(source_dir, target_dir, classes_to_include, use_copy=False):
    """
    Creates symbolic links (or copies) for specified classes from source to target directory.

    Args:
        source_dir (str): Path to the full ImageNet directory (e.g., './data').
        target_dir (str): Path to the target subset directory (e.g., './data_10_classes').
        classes_to_include (list): List of WNID strings for the classes to include.
        use_copy (bool): If True, copies files instead of creating symlinks.
    """
    print(f"\n--- Creating subset in: {target_dir} ---")
    print(f"Including {len(classes_to_include)} classes.")
    print(f"Using {'copy' if use_copy else 'symlink'} method.")

    source_train = Path(source_dir) / 'train'
    source_val = Path(source_dir) / 'val'
    target_train = Path(target_dir) / 'train'
    target_val = Path(target_dir) / 'val'

    if not source_train.is_dir():
        print(f"ERROR: Source train directory missing: {source_train}")
        return
    if not source_val.is_dir():
        print(f"ERROR: Source validation directory missing: {source_val}")
        print("Ensure you have run the 'prepare_val.py' script first.")
        return

    # Create target directories
    print("Creating target directories...")
    target_train.mkdir(parents=True, exist_ok=True)
    target_val.mkdir(parents=True, exist_ok=True)

    processed_classes = 0
    # Process both train and validation sets
    for split, source_split_dir, target_split_dir in [('train', source_train, target_train),
                                                      ('val', source_val, target_val)]:
        print(f"\nProcessing '{split}' split...")
        split_processed_classes = 0
        for class_wnid in classes_to_include:
            source_class_dir = source_split_dir / class_wnid
            target_class_dir = target_split_dir / class_wnid

            if not source_class_dir.is_dir():
                print(f"Warning: Source class directory not found for '{class_wnid}' in '{split}' split. Skipping: {source_class_dir}")
                continue

            # Create target class directory
            target_class_dir.mkdir(exist_ok=True)

            # Link or copy images
            try:
                image_files = list(source_class_dir.glob('*.JPEG'))
                if not image_files:
                    print(f"Warning: No JPEG images found in {source_class_dir}")
                    continue

                for src_img_path in image_files:
                    dest_img_path = target_class_dir / src_img_path.name
                    
                    # Use absolute path for source in symlink for robustness
                    src_abs_path = src_img_path.resolve() 

                    try:
                        if use_copy:
                             # Avoid re-copying if it exists
                            if not dest_img_path.exists():
                                shutil.copy2(src_abs_path, dest_img_path) # copy2 preserves metadata
                        else:
                            # Avoid creating link if it exists
                            if not dest_img_path.exists() and not dest_img_path.is_symlink():
                                os.symlink(src_abs_path, dest_img_path)
                    except FileExistsError:
                         # Should be caught by the 'if not exists' checks, but handle just in case
                         pass 
                    except OSError as e:
                        print(f"ERROR: Failed to {'copy' if use_copy else 'link'} {src_abs_path} to {dest_img_path}: {e}")
                
                split_processed_classes += 1

            except Exception as e:
                 print(f"ERROR processing class {class_wnid} in {split} split: {e}")

        print(f"Finished processing {split_processed_classes}/{len(classes_to_include)} classes for '{split}' split.")
        if split == 'train':
            processed_classes = split_processed_classes # Use train count as reference

    print(f"\n--- Subset creation finished for: {target_dir} ---")
    print(f"Successfully processed {processed_classes} classes (check warnings for skips).")

def main():
    parser = argparse.ArgumentParser(description="Create ImageNet subsets (10/100 classes) using symlinks or copies.")
    parser.add_argument('--source_dir', type=str, default='./data',
                        help="Path to the directory containing full ImageNet 'train' and 'val' folders.")
    parser.add_argument('--target_dir_10', type=str, default='./data_10_classes',
                        help="Path to the target directory for the 10-class subset.")
    parser.add_argument('--target_dir_100', type=str, default='./data_100_classes',
                        help="Path to the target directory for the 100-class subset.")
    parser.add_argument('--copy', action='store_true',
                        help="Use file copying instead of symbolic links (requires more disk space).")
    parser.add_argument('--classes_file', type=str, default=None,
                         help="Optional: Path to a file containing WNIDs (one per line) to define the classes, instead of taking the first N.")
    parser.add_argument('--num_classes_10', type=int, default=10, help="Number of classes for the first subset.")
    parser.add_argument('--num_classes_100', type=int, default=100, help="Number of classes for the second subset.")


    args = parser.parse_args()

    print("Starting ImageNet subset preparation...")
    print(f"Source ImageNet directory: {args.source_dir}")

    # --- Get Class List ---
    source_train_dir = os.path.join(args.source_dir, 'train')
    
    if args.classes_file:
        print(f"Reading class list from: {args.classes_file}")
        try:
            with open(args.classes_file, 'r') as f:
                all_classes = [line.strip() for line in f if line.strip()]
            print(f"Read {len(all_classes)} classes from file.")
            if len(all_classes) < max(args.num_classes_10, args.num_classes_100):
                 print(f"Warning: File contains fewer classes ({len(all_classes)}) than requested for subsets ({args.num_classes_10}/{args.num_classes_100}). Using all classes from file.")
                 args.num_classes_10 = min(args.num_classes_10, len(all_classes))
                 args.num_classes_100 = min(args.num_classes_100, len(all_classes))
        except FileNotFoundError:
            print(f"ERROR: Classes file not found: {args.classes_file}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to read classes file {args.classes_file}: {e}")
            sys.exit(1)
    else:
        print("Detecting classes from source train directory...")
        all_classes = get_imagenet_classes(source_train_dir)
        if all_classes is None:
            print("Could not determine ImageNet classes. Exiting.")
            sys.exit(1)
        if len(all_classes) < 1000:
             print(f"Warning: Found only {len(all_classes)} classes, expected 1000.")

    if not all_classes:
        print("No classes found or determined. Exiting.")
        sys.exit(1)
        
    # Select classes for subsets
    selected_classes_10 = all_classes[:args.num_classes_10]
    selected_classes_100 = all_classes[:args.num_classes_100]

    # --- Create Subsets ---
    # Create 10-class subset
    if args.num_classes_10 > 0:
        create_subset_links(args.source_dir, args.target_dir_10, selected_classes_10, args.copy)
    else:
        print("\nSkipping 10-class subset creation (num_classes_10 <= 0).")

    # Create 100-class subset
    if args.num_classes_100 > 0:
        create_subset_links(args.source_dir, args.target_dir_100, selected_classes_100, args.copy)
    else:
         print("\nSkipping 100-class subset creation (num_classes_100 <= 0).")

    print("\nSubset preparation script finished.")

if __name__ == "__main__":
    main()