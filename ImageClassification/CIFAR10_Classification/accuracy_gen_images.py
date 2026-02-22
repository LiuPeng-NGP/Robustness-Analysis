import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict # Import OrderedDict

# Assuming resnet.py contains the ResNet18 definition
try:
    from resnet import ResNet18
except ImportError:
    print("Error: Could not import ResNet18 from resnet.py.")
    print("Please ensure resnet.py is in the same directory or Python path.")
    exit()

# Assuming utils.py contains the progress_bar function
try:
    from utils import progress_bar
except ImportError:
    print("Warning: Could not import progress_bar from utils.py.")
    print("Progress bar will not be shown.")
    # Define a dummy progress_bar if utils.py is missing
    def progress_bar(batch_idx, total_batches, text):
        if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
             print(f"Batch {batch_idx+1}/{total_batches} | {text}")


# --- Argument Parser ---
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing on Custom Dataset')
parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing PNG images (e.g., 0.png, 1.png, ..., 10.png, ...)')
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the pre-trained checkpoint file (.pth)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for testing')
args = parser.parse_args()

# --- Device Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Data Loading ---
print('==> Preparing data..')

# Standard CIFAR10 Normalization values
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.filenames = [] # Keep track of filenames for potential debugging

        if not os.path.isdir(root_dir):
             print(f"Error: Image folder not found at '{root_dir}'")
             exit()

        print(f"Loading images from: {root_dir}")
        loaded_files = 0
        # Load images and labels
        for filename in sorted(os.listdir(root_dir)): # Sort for consistency
            if filename.lower().endswith(".png"): # Case-insensitive check
                try:
                    # Extract the number from the filename (before the first '.')
                    number_str = filename.split('.')[0]
                    number = int(number_str)
                    # Determine the class (number % 10 maps to CIFAR10 classes 0-9)
                    label = number % 10
                    # Load the image
                    img_path = os.path.join(root_dir, filename)
                    img = Image.open(img_path).convert('RGB') # Ensure RGB

                    # Basic check for image size (optional but good practice)
                    if img.size != (32, 32):
                        print(f"Warning: Image {filename} is not 32x32 ({img.size}). Resizing might occur depending on model, or errors might happen if not handled by transforms.")
                        # If your model expects exactly 32x32, you might add:
                        # img = img.resize((32, 32))

                    self.images.append(img)
                    self.labels.append(label)
                    self.filenames.append(filename)
                    loaded_files += 1
                except ValueError:
                    print(f"Warning: Could not parse number from filename: '{filename}'. Skipping.")
                except FileNotFoundError:
                     print(f"Warning: File suddenly disappeared: '{filename}'. Skipping.")
                except Exception as e:
                    print(f"Warning: Could not load or process image '{filename}': {e}. Skipping.")

        print(f"Loaded {loaded_files} images.")
        if loaded_files == 0:
            print(f"Error: No valid PNG images found or loaded from '{root_dir}'. Ensure filenames are like '0.png', '1.png', etc., and images are valid.")
            exit()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # filename = self.filenames[idx] # For debugging if needed
        if self.transform:
            image = self.transform(image)
        return image, label

# Create Dataset and DataLoader
testset = CustomImageDataset(root_dir=args.folder_path, transform=transform_test)

# Adjust batch size if dataset is smaller than default
actual_batch_size = min(args.batch_size, len(testset)) if len(testset) > 0 else 1
if actual_batch_size <= 0:
    print("Error: Cannot create DataLoader with batch size 0 (dataset is empty).")
    exit()

testloader = DataLoader(testset, batch_size=actual_batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# --- Model Loading ---
print('==> Building model..')
net = ResNet18()
net = net.to(device) # Move model to device first

# Load Checkpoint
print(f"==> Loading checkpoint '{args.checkpoint_path}'")
if not os.path.isfile(args.checkpoint_path):
    print(f"Error: Checkpoint file not found at '{args.checkpoint_path}'")
    exit()

# Use map_location for flexibility (GPU-trained model on CPU or vice-versa)
checkpoint = torch.load(args.checkpoint_path, map_location=device)

# --- Handle potential 'module.' prefix from DataParallel saving ---
state_dict = checkpoint.get('net', checkpoint) # Handle checkpoints that save the whole dict or just state_dict['net']
new_state_dict = OrderedDict()
is_data_parallel = False
for k, v in state_dict.items():
    if k.startswith('module.'):
        is_data_parallel = True
        name = k[7:] # remove `module.` prefix
        new_state_dict[name] = v
    else:
        new_state_dict[k] = v

# Load the potentially modified state dictionary
try:
    net.load_state_dict(new_state_dict)
except RuntimeError as e:
    print("\nError loading state_dict:")
    print(e)
    print("\nThis might happen if the model architecture in resnet.py doesn't match the architecture")
    print("used to save the checkpoint, or if the checkpoint is corrupted.")
    # Optionally print keys for debugging:
    # print("\nModel keys:")
    # print(list(net.state_dict().keys()))
    # print("\nCheckpoint keys (after potential prefix removal):")
    # print(list(new_state_dict.keys()))
    exit()

print("Checkpoint loaded successfully.")

# If device is CUDA and the model wasn't saved with DataParallel,
# but you have multiple GPUs, you might still want to wrap it.
# However, for *testing*, it often doesn't provide a huge speedup unless
# the dataset is massive, and adds complexity. We'll keep it simple here.
# If you NEED DataParallel for testing (e.g., memory reasons on multi-GPU):
if device == 'cuda' and torch.cuda.device_count() > 1:
     print("Multiple GPUs detected. Wrapping model with DataParallel for testing.")
     net = torch.nn.DataParallel(net)
     cudnn.benchmark = True # Enable benchmark for potential speedup

# --- Evaluation Function ---
criterion = nn.CrossEntropyLoss()

def test():
    net.eval() # Set model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0
    print("\nStarting evaluation...")
    with torch.no_grad(): # Disable gradient calculation for testing
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Use progress bar if available
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # --- Print Results ---
    if total > 0:
        final_acc = 100. * correct / total
        final_loss = test_loss / len(testloader)
        print("\n--- Evaluation Complete ---")
        print(f"Dataset Path: {args.folder_path}")
        print(f"Checkpoint Path: {args.checkpoint_path}")
        print(f"Total Images: {total}")
        print(f"Correct Predictions: {correct}")
        print(f"Average Loss: {final_loss:.4f}")
        print(f"Accuracy: {final_acc:.2f}%")
        print("--------------------------")
    else:
        print("\nEvaluation finished, but no samples were processed (dataset might be empty or filtered).")
        final_acc = 0.0

    return final_acc

# --- Run Testing ---
if len(testloader) > 0:
    test_acc = test()
else:
    print("Skipping evaluation as the test data loader is empty.")