cd ImageClassification/CIFAR10_Classification
# No wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:52650 train_ddp.py
Test Loss: 2.993 | Test Acc: 93.850% (9385/10000)

# 10% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:50121 train_ddp.py --wrong-rate 0.09090909 --checkpoint-name ckpt10p.pth --epochs 220
Test Loss: 3.763 | Test Acc: 94.080% (9408/10000)

# 30% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:50122 train_ddp.py --wrong-rate 0.23076923 --checkpoint-name ckpt30p.pth --epochs 260                                                          
Test Loss: 6.322 | Test Acc: 92.110% (9211/10000)

# 50% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:50123 train_ddp.py --wrong-rate 0.33333333 --checkpoint-name ckpt50p.pth --epochs 300
Test Loss: 8.817 | Test Acc: 87.960% (8796/10000)

# 100% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:50124 train_ddp.py --wrong-rate 0.5 --checkpoint-name ckpt100p.pth --epochs 400
Test Loss: 12.570 | Test Acc: 86.280% (8628/10000)