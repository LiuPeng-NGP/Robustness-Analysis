import torch
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from resnet import ResNet18
from scipy.stats import entropy

def test(net, testloader, device):
    """Function to test the network and return predictions and true labels."""
    net.eval()
    all_predicted = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return np.array(all_predicted), np.array(all_targets)

def calculate_distribution_metrics(predicted_labels, true_labels, num_classes):
    """Calculates metrics comparing the distributions of predicted and true labels."""
    # Get counts for each class using bincount for efficiency
    true_counts = np.bincount(true_labels, minlength=num_classes)
    pred_counts = np.bincount(predicted_labels, minlength=num_classes)

    # Convert counts to probability distributions
    p_true = true_counts / true_counts.sum()
    p_pred = pred_counts / pred_counts.sum()
    
    # Add a small epsilon to avoid division by zero or log(0) in KL divergence
    epsilon = 1e-9
    p_pred_smooth = p_pred + epsilon
    p_true_smooth = p_true + epsilon

    # Calculate metrics
    kl_divergence = entropy(pk=p_pred_smooth, qk=p_true_smooth)
    total_variation_dist = 0.5 * np.sum(np.abs(p_pred - p_true))

    return kl_divergence, total_variation_dist, true_counts, pred_counts

def analyze_label_distribution(predicted_labels, true_labels, num_classes, checkpoint_name):
    """Analyzes and plots the distribution of predicted and true labels."""
    plt.figure(figsize=(14, 7))
    plt.suptitle(f"Label Distribution Analysis for: {checkpoint_name}", fontsize=16)

    # Distribution of True Labels
    plt.subplot(1, 2, 1)
    sns.histplot(true_labels, bins=np.arange(num_classes + 1) - 0.5, shrink=0.8)
    plt.title('True Label Distribution (Test Set)')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(range(num_classes))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Distribution of Predicted Labels
    plt.subplot(1, 2, 2)
    sns.histplot(predicted_labels, bins=np.arange(num_classes + 1) - 0.5, shrink=0.8, color='orange')
    plt.title('Predicted Label Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(range(num_classes))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Label Distribution Analysis')
    parser.add_argument('--checkpoint-name', default='ckptddp.pth', type=str, help='checkpoint name to load')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = len(classes)

    # Model
    print('==> Building model..')
    net = ResNet18()
    net = net.to(device)

    # Load checkpoint
    checkpoint_path = os.path.join('./checkpoint', args.checkpoint_name)
    print(f'==> Loading model from {checkpoint_path}')
    assert os.path.isfile(checkpoint_path), f'Error: checkpoint file not found at {checkpoint_path}'
    
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    
    predicted_labels, true_labels = test(net, testloader, device)
    
    # Calculate and report distribution alignment metrics
    kl_div, tvd, true_counts, pred_counts = calculate_distribution_metrics(
        predicted_labels, true_labels, num_classes
    )

    print("\n--- Distribution Alignment Analysis ---")
    print(f"Checkpoint: {args.checkpoint_name}")
    print("\nClass Counts (Predicted vs. True):")
    for i, class_name in enumerate(classes):
        print(f"  Class {i} ({class_name:<5}): {pred_counts[i]:>5} vs. {true_counts[i]:>5}")

    print("\nDistribution Similarity Metrics (Lower is Better):")
    print(f"  Kullback-Leibler (KL) Divergence: {kl_div:.6f}")
    print(f"  Total Variation Distance (TVD):   {tvd:.6f}")
    print("-------------------------------------\n")
    
    # Analyze and plot the label distributions
    analyze_label_distribution(predicted_labels, true_labels, num_classes, args.checkpoint_name)

if __name__ == '__main__':
    main()