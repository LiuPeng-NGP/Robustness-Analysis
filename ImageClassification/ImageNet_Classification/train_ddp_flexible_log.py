#!/usr/bin/env python3
'''Train ImageNet Subset with PyTorch using ImageFolder - IMPROVED version with AdamW, Warmup, RandAugment, Label Smoothing etc.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.transforms as transforms
# Use ImageFolder for both training and validation for subsets
from torchvision.datasets import ImageFolder # <--- Changed from ImageNet

import os
import argparse
import time
import logging # Added for logging

# Import OrderedDict from collections
from collections import OrderedDict

# Import the Vision Transformer model
from vit import ViT_B_16 # Example: using ViT-Base with 16x16 patches

# --- Global logger instance ---
# Logger will be configured by setup_logging() after DDP init
logger = logging.getLogger(__name__)


# Assuming utils.py contains AverageMeter and ProgressMeter (or similar)
# If not, you'll need to define them or import them appropriately.
# Placeholder definitions if utils.py is not available:
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

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if torch.cuda.is_available(): # Add memory usage if available
             # 'device' should be accessible in the scope where ProgressMeter instances are used
             # or passed if ProgressMeter were in a separate utility file.
             # Here, 'device' is a global variable set up during DDP initialization.
             entries.append(f"Mem: {torch.cuda.memory_allocated(device)/1e9:.2f}G/{torch.cuda.max_memory_allocated(device)/1e9:.2f}G")
        # MODIFIED: Use logger.info instead of print. Assumes 'logger' is the global logger instance.
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
# --- End Placeholder utils ---

# Import numpy for label flipping
import numpy as np

# Import RandAugment
from timm.data.auto_augment import rand_augment_transform

parser = argparse.ArgumentParser(description='PyTorch ImageNet Subset Training using ViT - Improved')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for AdamW')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint-name', default='vit_imagenet10_ddp_improved.pth', type=str, help='checkpoint name')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs to train (adjust for subset)')
parser.add_argument('--data-path', default='./data_10_classes', type=str, help='path to ImageNet SUBSET dataset')
parser.add_argument('--batch-size-per-gpu', default=128, type=int, help='batch size per GPU')
parser.add_argument('--num-workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--model', default='ViT_B_16', type=str, help='ViT model variant (e.g., ViT_Ti_16, ViT_S_16, ViT_B_16, ViT_L_16)')
parser.add_argument('--num-classes', default=100, type=int, help='Number of classes in the dataset subset')
parser.add_argument('--warmup-epochs', default=5, type=int, help='number of warmup epochs')
parser.add_argument('--weight-decay', default=0.05, type=float, help='weight decay for AdamW')
parser.add_argument('--label-smoothing', type=float, default=0.1, help='label smoothing factor (can be reduced/removed for smaller datasets)')
parser.add_argument('--drop-path', type=float, default=0.1, help='stochastic depth rate')
parser.add_argument('--clip-grad', type=float, default=1.0, help='gradient clipping max norm')
parser.add_argument('--wrong-rate', default=0.0, type=float, help='Rate at which training labels are randomly flipped (0.0 to 1.0)')
# Added log file base name argument
parser.add_argument('--log-file-base', default='training_vit_imagenet_improved', type=str, help='Base name for the log file (rank 0 will append .log)')
# Added log directory argument
parser.add_argument('--log-dir', default='./logs', type=str, help='Directory to save log files')


args = parser.parse_args()

# --- Logging Setup Function ---
def setup_logging(log_dir=".", log_file_name_base="training_script", local_rank=0, world_size=1, log_level=logging.INFO):
    """
    Sets up logging.
    - All ranks log to console (StreamHandler) with rank information.
    - Only rank 0 logs to a file (FileHandler) without rank information in message (rank is implied).
    """
    _logger = logger
    _logger.setLevel(log_level)

    if _logger.hasHandlers():
        _logger.handlers.clear()

    console_formatter_str = f"[%(asctime)s][Rank {local_rank}/{world_size}][%(levelname)s] %(message)s"
    console_formatter = logging.Formatter(console_formatter_str)
    ch = logging.StreamHandler()
    ch.setFormatter(console_formatter)
    _logger.addHandler(ch)

    if local_rank == 0:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        file_log_name = os.path.join(log_dir, f"{log_file_name_base}.log")
        file_formatter_str = f"[%(asctime)s][%(levelname)s] %(message)s"
        file_formatter = logging.Formatter(file_formatter_str)
        
        fh = logging.FileHandler(file_log_name, mode='a') # Append mode
        fh.setFormatter(file_formatter)
        _logger.addHandler(fh)
        _logger.info(f"File logging to {file_log_name} initialized.")

    _logger.propagate = False


# --- Function to potentially flip labels ---
def maybe_flip_labels(c, flip_rate, num_classes):
    if num_classes <= 1: return c
    if np.random.rand() < flip_rate:
        possible_labels = list(range(num_classes))
        if c in possible_labels: possible_labels.remove(c)
        if not possible_labels: return c
        return np.random.choice(possible_labels)
    return c

# Initialize distributed training
rank = 0
world_size = 1
local_rank = 0
device = 'cpu'

if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Pass args.log_dir to setup_logging
    setup_logging(log_dir=args.log_dir, log_file_name_base=args.log_file_base, local_rank=local_rank, world_size=world_size)
    logger.info(f"DDP Setup: RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        device = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        logger.info(f"Running DDP on cuda:{device}.")
    else:
        device = 'cpu'
        logger.warning("NCCL backend specified but CUDA not available. DDP may fail. Running on CPU.")
    
    local_seed = 1334 + local_rank
    logger.info(f"Seeding with {local_seed} for rank {local_rank}.")
else:
    # Pass args.log_dir to setup_logging
    setup_logging(log_dir=args.log_dir, log_file_name_base=args.log_file_base, local_rank=local_rank, world_size=world_size)
    logger.warning("Not using DDP from environment variables. Assuming single process or manual DDP setup.")
    
    if not dist.is_initialized():
        try:
            dist.init_process_group(backend='gloo', init_method='file:///tmp/ddp_train_sharedfile_unique', rank=0, world_size=1)
            logger.info("Initialized 'gloo' process group for single process.")
        except Exception as e:
            logger.warning(f"Could not initialize 'gloo' process group for single process: {e}. Some DDP functionalities might not work as expected.")

    if torch.cuda.is_available():
        device = 0
        torch.cuda.set_device(device)
        logger.info(f"Running single process on cuda:{device}.")
    else:
        device = 'cpu'
        logger.warning("CUDA not available, running single process on CPU. This will be very slow.")
    
    local_seed = 1334
    logger.info(f"Seeding with {local_seed} for single process.")


if local_rank == 0:
    logger.info(f"Script arguments: {args}")

torch.manual_seed(local_seed)
np.random.seed(local_seed)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

best_acc = 0
start_epoch = 0

# Data
if local_rank == 0:
    logger.info(f'==> Preparing ImageNet subset ({args.num_classes} classes) data from {args.data_path}..')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img_size = 224
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    rand_augment_transform(config_str='rand-m9-n2-mstd0.5', hparams={'translate_const': int(img_size * 0.45)}),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    normalize,
])

traindir = os.path.join(args.data_path, 'train')
valdir = os.path.join(args.data_path, 'val')

if not os.path.isdir(traindir):
    err_msg = f"ImageNet subset train directory not found at {traindir}. Check --data-path."
    logger.error(err_msg)
    raise FileNotFoundError(err_msg)
if not os.path.isdir(valdir):
    err_msg = f"ImageNet subset validation directory not found at {valdir}. Check --data-path."
    logger.error(err_msg)
    raise FileNotFoundError(err_msg)

if local_rank == 0:
    logger.info(f"Using ImageFolder for training data from: {traindir}")
    logger.info(f"Using ImageFolder for validation data from: {valdir}")
    if args.wrong_rate > 0:
        logger.info(f"Applying label flipping during training with rate: {args.wrong_rate}")

trainset = ImageFolder(root=traindir, transform=transform_train)
testset = ImageFolder(root=valdir, transform=transform_test)

detected_classes = len(trainset.classes)
if detected_classes != args.num_classes:
    if local_rank == 0:
        logger.warning(f"Detected {detected_classes} classes in {traindir}, but --num-classes was set to {args.num_classes}. Using {detected_classes} classes.")
    args.num_classes = detected_classes
elif local_rank == 0:
    logger.info(f"Successfully found {args.num_classes} classes in training directory.")


train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=local_rank, shuffle=True)
total_batch_size = args.batch_size_per_gpu * world_size
if local_rank == 0:
    logger.info(f"Using Total Batch Size: {total_batch_size} ({args.batch_size_per_gpu} per GPU)")

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size_per_gpu, shuffle=False,
    num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=local_rank, shuffle=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size_per_gpu, shuffle=False,
    num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)

if local_rank == 0:
    logger.info(f'==> Building model {args.model} for {args.num_classes} classes..')
try:
    if args.model == 'ViT_Ti_16': from vit import ViT_Ti_16; net_builder = ViT_Ti_16
    elif args.model == 'ViT_S_16': from vit import ViT_S_16; net_builder = ViT_S_16
    elif args.model == 'ViT_L_16': from vit import ViT_L_16; net_builder = ViT_L_16
    elif args.model == 'ViT_B_16': from vit import ViT_B_16; net_builder = ViT_B_16
    else: raise ImportError(f"Model {args.model} not found in vit.py or not supported.")
    net = net_builder(img_size=img_size, num_classes=args.num_classes, drop_path_rate=args.drop_path)
except ImportError as e:
    logger.error(f"Error importing ViT model: {e}")
    logger.error("Please ensure vit.py is in the same directory or your Python path and contains the specified model.")
    exit(1)
except NameError as e:
    logger.error(f"Error: ViT model class '{args.model}' not defined after import: {e}")
    logger.error("Please check the class name in vit.py.")
    exit(1)

net = net.to(device)
if world_size > 1 and torch.cuda.is_available() and dist.is_initialized():
    net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
elif world_size > 1 and device == 'cpu' and dist.is_initialized():
    net = DDP(net, find_unused_parameters=False) 
elif world_size == 1 and dist.is_initialized():
     net = DDP(net, find_unused_parameters=False)


def count_parameters(model_to_count):
    return sum(p.numel() for p in model_to_count.parameters() if p.requires_grad)
if local_rank == 0:
    model_to_inspect = net.module if isinstance(net, DDP) else net
    num_params = count_parameters(model_to_inspect)
    logger.info(f'Number of trainable parameters: {num_params / 1e6:.2f}M')

criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))
scaler = GradScaler(enabled=torch.cuda.is_available())

def train(epoch):
    if local_rank == 0:
        logger.info(f'\nEpoch: {epoch}')
    net.train()
    if isinstance(train_sampler, DistributedSampler):
        train_sampler.set_epoch(epoch)
    
    train_loss = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    progress = ProgressMeter(
        len(trainloader),
        [batch_time, data_time, train_loss, top1],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        if args.wrong_rate > 0:
            targets_flipped = torch.tensor(
                [maybe_flip_labels(label.item(), flip_rate=args.wrong_rate, num_classes=args.num_classes)
                 for label in targets], dtype=targets.dtype, device=device)
        else:
            targets_flipped = targets

        is_warmup = epoch < args.warmup_epochs
        if is_warmup and args.warmup_epochs > 0:
            current_step = batch_idx + epoch * len(trainloader)
            total_warmup_steps = args.warmup_epochs * len(trainloader)
            warmup_factor = min(1.0, current_step / total_warmup_steps) if total_warmup_steps > 0 else 1.0
            base_lr = args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * warmup_factor
        
        with autocast(enabled=torch.cuda.is_available()):
            outputs = net(inputs)
            loss = criterion(outputs, targets_flipped)

        scaler.scale(loss).backward()
        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        acc1, = accuracy(outputs, targets, topk=(1,))
        
        if world_size > 1 and dist.is_initialized():
            reduced_loss = reduce_tensor(loss.data, world_size)
            reduced_acc1 = reduce_tensor(acc1, world_size)
        else:
            reduced_loss = loss.data
            reduced_acc1 = acc1
        
        train_loss.update(reduced_loss.item(), inputs.size(0))
        top1.update(reduced_acc1.item(), inputs.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()

        if local_rank == 0 and batch_idx % 50 == 0:
            progress.display(batch_idx)
            if is_warmup and args.warmup_epochs > 0:
                 current_lr = optimizer.param_groups[0]['lr']
                 logger.info(f"Warmup LR: {current_lr:.6f}")

    if local_rank == 0:
        logger.info(f"Epoch {epoch} Train Summary: Loss: {train_loss.avg:.3f} | Acc@1: {top1.avg:.3f}%")

def test(epoch):
    global best_acc
    net.eval()
    test_loss = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    batch_time = AverageMeter('Time', ':6.3f')

    progress = ProgressMeter(
        len(testloader),
        [batch_time, test_loss, top1],
        prefix='Test: ')

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with autocast(enabled=torch.cuda.is_available()):
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            acc1, = accuracy(outputs, targets, topk=(1,))
            
            if world_size > 1 and dist.is_initialized():
                reduced_loss = reduce_tensor(loss.data, world_size)
                reduced_acc1 = reduce_tensor(acc1, world_size)
            else:
                reduced_loss = loss.data
                reduced_acc1 = acc1

            test_loss.update(reduced_loss.item(), inputs.size(0))
            top1.update(reduced_acc1.item(), inputs.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()

            if local_rank == 0 and batch_idx % 50 == 0:
                 progress.display(batch_idx)

    if local_rank == 0:
        logger.info(f'* Test Acc@1 {top1.avg:.3f} Loss {test_loss.avg:.3f}')
        acc = top1.avg
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        if is_best:
            logger.info(f'Saving new best checkpoint with Acc@1: {best_acc:.3f}%')
            model_state_to_save = net.module.state_dict() if isinstance(net, DDP) else net.state_dict()
            state = {
                'net': model_state_to_save,
                'acc': best_acc, 'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler_cosine.state_dict(),
                'scaler': scaler.state_dict(), 'args': args
            }
            checkpoint_dir = './checkpoint'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(state, os.path.join(checkpoint_dir, args.checkpoint_name))

def reduce_tensor(tensor, n_procs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_procs
    return rt

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk); batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if args.resume:
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} if torch.cuda.is_available() and world_size > 1 else device
    checkpoint_path = os.path.join('./checkpoint', args.checkpoint_name)
    if os.path.isfile(checkpoint_path):
        if local_rank == 0: logger.info(f'==> Resuming from checkpoint {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        load_optimizer_etc = True 

        model_to_load = net.module if isinstance(net, DDP) else net

        if 'args' in checkpoint and hasattr(checkpoint['args'], 'num_classes'):
            ckpt_num_classes = checkpoint['args'].num_classes
            if ckpt_num_classes != args.num_classes:
                if local_rank == 0:
                    logger.warning(f"!!!!!!!! WARNING !!!!!!!!!")
                    logger.warning(f"Checkpoint was trained with {ckpt_num_classes} classes, but current run is configured for {args.num_classes} classes.")
                    logger.warning(f"Loading weights EXCEPT the final classification layer ('head').")
                    logger.warning(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                state_dict = checkpoint['net']
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
                model_to_load.load_state_dict(state_dict, strict=False)
                if local_rank == 0: logger.info(f"Loaded weights from checkpoint, excluding the classification head.")
                best_acc = 0; start_epoch = 0
                load_optimizer_etc = False
            else:
                 model_to_load.load_state_dict(checkpoint['net'])
        else:
            if local_rank == 0: logger.warning("Checkpoint args not found or missing num_classes. Attempting full load; head layer might mismatch if num_classes changed.")
            try:
                model_to_load.load_state_dict(checkpoint['net'])
            except RuntimeError as e:
                if local_rank == 0:
                    logger.error(f"Error loading state dict fully (likely due to num_classes mismatch or architecture change): {e}")
                    logger.info("Attempting partial load of matching parameters...")
                model_dict = model_to_load.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(pretrained_dict)
                model_to_load.load_state_dict(model_dict)
                if local_rank == 0: logger.info(f"Partially loaded {len(pretrained_dict)} matching parameters.")
                best_acc = 0; start_epoch = 0
                load_optimizer_etc = False
        
        if 'acc' in checkpoint and load_optimizer_etc: best_acc = checkpoint['acc']
        if 'epoch' in checkpoint and load_optimizer_etc: start_epoch = checkpoint['epoch'] + 1

        if load_optimizer_etc:
            try:
                if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint: scheduler_cosine.load_state_dict(checkpoint['scheduler'])
                if 'scaler' in checkpoint: scaler.load_state_dict(checkpoint['scaler'])
                if local_rank == 0: logger.info(f"Resumed optimizer, scheduler, and scaler states from epoch {checkpoint.get('epoch', 'N/A')} with accuracy {best_acc:.2f}%")
            except Exception as e:
                if local_rank == 0:
                    logger.warning(f"Warning: Could not load optimizer/scheduler/scaler state: {e}")
                    logger.warning(f"Resuming model weights from epoch {checkpoint.get('epoch', 'N/A')} with accuracy {best_acc:.2f}%. Optimizer/scheduler/scaler state might be missing or incompatible.")
    else:
        if local_rank == 0: logger.info(f'==> No checkpoint found at {checkpoint_path}, starting from scratch.')

for epoch in range(start_epoch, args.epochs):
    train(epoch)
    test(epoch)
    if epoch >= args.warmup_epochs and args.warmup_epochs < args.epochs :
        scheduler_cosine.step()
    if local_rank == 0:
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"End of Epoch {epoch} LR: {current_lr:.6f}")

if dist.is_initialized():
    dist.destroy_process_group()

if local_rank == 0:
    logger.info("Training Finished!")
    logger.info(f"Best Test Accuracy (Top-1): {best_acc:.3f}% for {args.num_classes} classes.")
    log_file_path = os.path.join(args.log_dir, f"{args.log_file_base}.log")
    logger.info(f"Checkpoints saved in ./checkpoint/. Log file: {log_file_path}")