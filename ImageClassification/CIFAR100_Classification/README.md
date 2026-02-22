cd ImageClassification/CIFAR100_Classification
# No wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51650 train_ddp.py                             
Test Loss: 12.966 | Test Acc: 75.930% (7593/10000)

# 10% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51121 train_ddp.py --wrong-rate 0.09090909 --checkpoint-name ckpt10p.pth --epochs 220                           
Test Loss: 14.617 | Test Acc: 76.440% (7644/10000)

# 30% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51122 train_ddp.py --wrong-rate 0.23076923 --checkpoint-name ckpt30p.pth --epochs 260                              
Test Loss: 18.238 | Test Acc: 72.340% (7234/10000)

# 50% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51123 train_ddp.py --wrong-rate 0.33333333 --checkpoint-name ckpt50p.pth --epochs 300                               
Test Loss: 21.911 | Test Acc: 66.160% (6616/10000)

# 100% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51124 train_ddp.py --wrong-rate 0.5 --checkpoint-name ckpt100p.pth --epochs 400                            
Test Loss: 26.296 | Test Acc: 61.000% (6100/10000)