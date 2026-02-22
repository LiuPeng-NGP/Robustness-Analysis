# utils.py
'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
    - AverageMeter: Computes and stores the average and current value.
'''
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    # Note: This might be very slow for ImageNet
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    num_samples = 0.
    for inputs, _ in dataloader:
        # Input shape [1, C, H, W]
        batch_samples = inputs.size(0)
        inputs = inputs.view(batch_samples, inputs.size(1), -1)
        mean += inputs.mean(2).sum(0)
        std += inputs.std(2).sum(0)
        num_samples += batch_samples
        # Add progress indication if desired, as this takes long
        if int(num_samples) % 10000 == 0:
            print(f"Processed {int(num_samples)} images...")

    mean /= num_samples
    std /= num_samples
    print(f'Mean: {mean}')
    print(f'Std: {std}')
    return mean, std

def init_params(net):
    '''Init layer parameters - More relevant for ConvNets.'''
    # Note: ViT models usually have their own specific initialization.
    # Applying this generic init might override beneficial ViT initialization.
    print("Warning: Applying generic init_params to a ViT model might not be optimal.")
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out') # Use underscore version
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3) # Might conflict with ViT's trunc_normal_
            if m.bias is not None:
                init.constant_(m.bias, 0)

# Get terminal width - handles potential errors if 'stty size' fails
try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except ValueError:
    term_width = 80 # Default width

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    # Ensure lengths are non-negative
    cur_len = max(0, cur_len)
    rest_len = max(0, rest_len)

    sys.stdout.write(' [')
    sys.stdout.write('=' * cur_len)
    sys.stdout.write('>')
    sys.stdout.write('.' * rest_len)
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append(' Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    # Calculate padding carefully to avoid errors with varying terminal widths/message lengths
    space_needed = term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3
    if space_needed > 0:
        sys.stdout.write(' ' * space_needed)

    # Go back to the center of the bar.
    # Calculate backspaces carefully
    backspaces = term_width - int(TOTAL_BAR_LENGTH/2) + 2
    if backspaces > 0:
        sys.stdout.write('\b' * backspaces)
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

# Optional: Add AverageMeter here instead of in train_ddp.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)