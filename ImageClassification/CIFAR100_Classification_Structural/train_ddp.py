'''Train CIFAR100 with PyTorch and systematic label flipping.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.transforms as transforms

import os
import argparse

# Assuming resnet.py contains a ResNet18 model definition.
# Note: The ResNet18 model must be adapted for 100 classes,
# e.g., by setting the number of output features in the final linear layer to 100.
from resnet import ResNet18

from utils import progress_bar
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--wrong-rate', default=0, type=float, help='wrong rate')
parser.add_argument('--checkpoint-name', default='ckptddp_cifar100.pth', type=str, help='checkpoint name')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs to train')
args = parser.parse_args()


def maybe_flip_labels(c, flip_rate, num_classes):
    """
    Decide whether to flip the label. If a flip occurs, the label is consistently
    changed to the next label in a circular fashion (e.g., 0->1, 1->2, ..., 99->0).
    """
    if np.random.rand() < flip_rate:
        # Consistently mislabel to the next label, with the last label mapping to the first.
        return (c + 1) % num_classes
    return c

# Setup DDP:
dist.init_process_group("nccl")
# The batch size of 256 is an example; adjust as needed.
# assert 256 % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
local_rank = dist.get_rank()
device = local_rank % torch.cuda.device_count()
local_seed = 1334 + local_rank
torch.cuda.set_device(device)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
train_sampler = DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=(train_sampler is None), num_workers=2, sampler=train_sampler)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
test_sampler = DistributedSampler(testset)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2, sampler=test_sampler)

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
net = DDP(net, device_ids=[local_rank])


# Calculate the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(net)
if dist.get_rank() == 0:
    print(f'Number of parameters: {num_params}')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scaler = GradScaler()

# Training
def train(epoch):
    if dist.get_rank() == 0:
        print('\nEpoch: %d' % epoch)
    net.train()
    train_sampler.set_epoch(epoch)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Apply systematic label flipping for CIFAR-100 (100 classes)
        targets_flipped = torch.tensor([maybe_flip_labels(label.item(), flip_rate=args.wrong_rate, num_classes=100) for label in targets], device=device)
        with autocast():
            outputs = net(inputs)
            loss = criterion(outputs, targets_flipped)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if dist.get_rank() == 0:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_sampler.set_epoch(epoch)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if dist.get_rank() == 0:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Reduce accuracy and loss across all processes
    reduced_loss = torch.tensor(test_loss).to(device)
    reduced_total = torch.tensor(total).to(device)
    reduced_correct = torch.tensor(correct).to(device)

    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(reduced_total, op=dist.ReduceOp.SUM)
    dist.all_reduce(reduced_correct, op=dist.ReduceOp.SUM)

    # Note: The loss is averaged over all processes, but the total and correct counts are summed.
    reduced_loss = reduced_loss.item() / dist.get_world_size()
    reduced_total = reduced_total.item()
    reduced_correct = reduced_correct.item()

    acc = 100. * reduced_correct / reduced_total
    if dist.get_rank() == 0:
        print(f'Test Loss: {reduced_loss:.3f} | Test Acc: {acc:.3f}% ({reduced_correct}/{reduced_total})')

    # Save checkpoint.
    if acc > best_acc and dist.get_rank() == 0:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join('./checkpoint', args.checkpoint_name))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()

# Clean up
dist.destroy_process_group()