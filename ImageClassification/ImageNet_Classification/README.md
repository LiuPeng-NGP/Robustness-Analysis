cd ImageClassification/ImageNet_Classification
conda activate data_quality

# Dataset
## 1000 classes
cd data
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz

mkdir train
mkdir val
tar xf ILSVRC2012_img_train.tar -C train/
cd train

for f in *.tar; do
  d="${f%.tar}" # Get directory name by removing .tar
  mkdir "$d"
  tar -xf "$f" -C "$d/"
  rm "$f" # Optional: remove the class .tar file after extraction
done

cd ..
tar xf ILSVRC2012_img_val.tar -C val

tar -xzf ILSVRC2012_devkit_t12.tar.gz
cd ..
python3 prepare_val.py

python3 prepare_10_100_classes.py



# Train
## 1000 classes
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53621 train_ddp_flexible_log.py --data-path data --num-classes 1000 --epochs 300 --log-file-base c1000e0 --checkpoint-name ckpt_1000_classes_0p.pth --wrong-rate 0  
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53622 train_ddp_flexible_log.py --data-path data --num-classes 1000 --epochs 330 --log-file-base c1000e10 --checkpoint-name ckpt_1000_classes_10p.pth --wrong-rate 0.09090909
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53623 train_ddp_flexible_log.py --data-path data --num-classes 1000 --epochs 390 --log-file-base c1000e30 --checkpoint-name ckpt_1000_classes_30p.pth --wrong-rate 0.23076923
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53624 train_ddp_flexible_log.py --data-path data --num-classes 1000 --epochs 450 --log-file-base c1000e50 --checkpoint-name ckpt_1000_classes_50p.pth --wrong-rate 0.33333333
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53625 train_ddp_flexible_log.py --data-path data --num-classes 1000 --epochs 600 --log-file-base c1000e100 --checkpoint-name ckpt_1000_classes_100p.pth --wrong-rate 0.5

### c1000e0
[2025-05-09 10:26:06,718][INFO] Epoch 299 Train Summary: Loss: 1.250 | Acc@1: 94.244%
[2025-05-09 10:26:08,359][INFO] Test: [ 0/49]	Time  1.640 ( 1.640)	Loss 1.4694e+00 (1.4694e+00)	Acc@1  89.36 ( 89.36)	Mem: 1.89G/14.79G
[2025-05-09 10:26:19,152][INFO] * Test Acc@1 73.706 Loss 2.193
[2025-05-09 10:26:19,153][INFO] End of Epoch 299 LR: 0.000000
[2025-05-09 10:26:21,574][INFO] Training Finished!
[2025-05-09 10:26:21,574][INFO] Best Test Accuracy (Top-1): 73.784% for 1000 classes.
[2025-05-09 10:26:21,575][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c1000e0.log

### c1000e10
[2025-05-09 15:53:23,765][INFO] Epoch 329 Train Summary: Loss: 1.960 | Acc@1: 93.528%
[2025-05-09 15:53:25,316][INFO] Test: [ 0/49]	Time  1.549 ( 1.549)	Loss 1.5041e+00 (1.5041e+00)	Acc@1  89.16 ( 89.16)	Mem: 1.89G/14.79G
[2025-05-09 15:53:32,380][INFO] * Test Acc@1 73.430 Loss 2.261
[2025-05-09 15:53:32,381][INFO] End of Epoch 329 LR: 0.000000
[2025-05-09 15:53:34,817][INFO] Training Finished!
[2025-05-09 15:53:34,817][INFO] Best Test Accuracy (Top-1): 73.530% for 1000 classes.
[2025-05-09 15:53:34,817][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c1000e10.log

[2025-05-09 11:17:05,667][INFO] Epoch 299 Train Summary: Loss: 1.987 | Acc@1: 92.728%
[2025-05-09 11:17:07,237][INFO] Test: [ 0/49]	Time  1.568 ( 1.568)	Loss 1.5196e+00 (1.5196e+00)	Acc@1  88.96 ( 88.96)	Mem: 1.89G/14.79G
[2025-05-09 11:17:14,286][INFO] * Test Acc@1 72.992 Loss 2.267
[2025-05-09 11:17:14,286][INFO] End of Epoch 299 LR: 0.000021
[2025-05-09 11:17:14,286][INFO] 


### c1000e30
[2025-05-10 16:50:48,131][INFO] Epoch 389 Train Summary: Loss: 2.949 | Acc@1: 92.984%
[2025-05-10 16:50:49,570][INFO] Test: [ 0/49]	Time  1.437 ( 1.437)	Loss 1.5633e+00 (1.5633e+00)	Acc@1  89.16 ( 89.16)	Mem: 1.89G/14.79G
[2025-05-10 16:50:54,156][INFO] * Test Acc@1 73.646 Loss 2.328
[2025-05-10 16:50:54,156][INFO] End of Epoch 389 LR: 0.000000
[2025-05-10 16:50:56,090][INFO] Training Finished!
[2025-05-10 16:50:56,090][INFO] Best Test Accuracy (Top-1): 73.728% for 1000 classes.
[2025-05-10 16:50:56,090][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c1000e30.log

[2025-05-10 10:26:40,078][INFO] Epoch 299 Train Summary: Loss: 3.121 | Acc@1: 87.162%
[2025-05-10 10:26:41,485][INFO] Test: [ 0/49]	Time  1.405 ( 1.405)	Loss 1.5716e+00 (1.5716e+00)	Acc@1  88.67 ( 88.67)	Mem: 1.89G/14.79G
[2025-05-10 10:26:46,086][INFO] * Test Acc@1 72.302 Loss 2.266
[2025-05-10 10:26:46,086][INFO] End of Epoch 299 LR: 0.000129
[2025-05-10 10:26:46,086][INFO] 


### c1000e50
[2025-05-09 10:56:44,259][INFO] Epoch 449 Train Summary: Loss: 3.624 | Acc@1: 92.498%
[2025-05-09 10:56:45,807][INFO] Test: [ 0/49]	Time  1.546 ( 1.546)	Loss 1.6951e+00 (1.6951e+00)	Acc@1  88.57 ( 88.57)	Mem: 1.89G/14.79G
[2025-05-09 10:56:50,354][INFO] * Test Acc@1 73.684 Loss 2.398
[2025-05-09 10:56:50,355][INFO] End of Epoch 449 LR: 0.000000
[2025-05-09 10:56:52,405][INFO] Training Finished!
[2025-05-09 10:56:52,405][INFO] Best Test Accuracy (Top-1): 73.700% for 1000 classes.
[2025-05-09 10:56:52,405][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c1000e50.log

[2025-05-09 00:10:36,839][INFO] Epoch 299 Train Summary: Loss: 3.920 | Acc@1: 80.533%
[2025-05-09 00:10:38,249][INFO] Test: [ 0/49]	Time  1.408 ( 1.408)	Loss 1.6805e+00 (1.6805e+00)	Acc@1  87.30 ( 87.30)	Mem: 1.89G/14.79G
[2025-05-09 00:10:42,987][INFO] * Test Acc@1 71.854 Loss 2.313
[2025-05-09 00:10:42,988][INFO] End of Epoch 299 LR: 0.000255
[2025-05-09 00:10:42,988][INFO] 


### c1000e100
[2025-05-12 05:50:27,758][INFO] Epoch 599 Train Summary: Loss: 4.646 | Acc@1: 91.431%
[2025-05-12 05:50:29,175][INFO] Test: [ 0/49]	Time  1.416 ( 1.416)	Loss 1.8324e+00 (1.8324e+00)	Acc@1  90.92 ( 90.92)	Mem: 1.89G/14.79G
[2025-05-12 05:50:33,814][INFO] * Test Acc@1 74.722 Loss 2.506
[2025-05-12 05:50:33,815][INFO] End of Epoch 599 LR: 0.000000
[2025-05-12 05:50:35,672][INFO] Training Finished!
[2025-05-12 05:50:35,673][INFO] Best Test Accuracy (Top-1): 74.778% for 1000 classes.
[2025-05-12 05:50:35,673][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c1000e100.log

[2025-05-08 22:42:20,430][INFO] Epoch 299 Train Summary: Loss: 5.092 | Acc@1: 66.870%
[2025-05-08 22:42:21,964][INFO] Test: [ 0/56]	Time  1.532 ( 1.532)	Loss 1.9417e+00 (1.9417e+00)	Acc@1  88.28 ( 88.28)	Mem: 2.23G/15.13G
[2025-05-08 22:42:27,318][INFO] Test: [50/56]	Time  0.064 ( 0.135)	Loss 2.9556e+00 (2.5137e+00)	Acc@1  59.04 ( 70.87)	Mem: 2.23G/15.13G
[2025-05-08 22:42:27,754][INFO] * Test Acc@1 71.093 Loss 2.505
[2025-05-08 22:42:27,755][INFO] End of Epoch 299 LR: 0.000509
[2025-05-08 22:42:27,755][INFO] 




## 100 classes
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53611 train_ddp_flexible_log.py --data-path data_100_classes --num-classes 100 --log-file-base c100e0 --epochs 300 --checkpoint-name ckpt_100_classes_0p.pth --wrong-rate 0  
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53612 train_ddp_flexible_log.py --data-path data_100_classes --num-classes 100 --log-file-base c100e10 --epochs 330 --checkpoint-name ckpt_100_classes_10p.pth --wrong-rate 0.09090909
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53613 train_ddp_flexible_log.py --data-path data_100_classes --num-classes 100 --log-file-base c100e30 --epochs 195 --batch-size-per-gpu 256 --checkpoint-name ckpt_100_classes_30p.pth --wrong-rate 0.23076923
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53614 train_ddp_flexible_log.py --data-path data_100_classes --num-classes 100 --log-file-base c100e50 --epochs 275 --batch-size-per-gpu 256 --checkpoint-name ckpt_100_classes_50p.pth --wrong-rate 0.33333333
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53615 train_ddp_flexible_log.py --data-path data_100_classes --num-classes 100 --log-file-base c100e100 --epochs 150 --batch-size-per-gpu 512 --checkpoint-name ckpt_100_classes_100p.pth --wrong-rate 0.5

### c100e0
Epoch: 299
[2025-05-07 20:00:18,106][INFO] Epoch: [299][  0/127]	Time  1.982 ( 1.982)	Data  1.117 ( 1.117)	Loss 1.1165e+00 (1.1165e+00)	Acc@1  90.04 ( 90.04)	Mem: 1.87G/14.78G
[2025-05-07 20:00:45,798][INFO] Epoch: [299][ 50/127]	Time  0.660 ( 0.582)	Data  0.001 ( 0.022)	Loss 1.0828e+00 (1.0937e+00)	Acc@1  92.19 ( 90.97)	Mem: 1.87G/14.78G
[2025-05-07 20:01:19,170][INFO] Epoch: [299][100/127]	Time  0.666 ( 0.624)	Data  0.000 ( 0.012)	Loss 1.1099e+00 (1.0943e+00)	Acc@1  90.04 ( 90.89)	Mem: 1.87G/14.78G
[2025-05-07 20:01:36,406][INFO] Epoch 299 Train Summary: Loss: 1.096 | Acc@1: 90.851%
[2025-05-07 20:01:37,926][INFO] Test: [0/5]	Time  1.518 ( 1.518)	Loss 1.6071e+00 (1.6071e+00)	Acc@1  75.98 ( 75.98)	Mem: 1.87G/14.78G
[2025-05-07 20:01:38,861][INFO] * Test Acc@1 63.420 Loss 2.166
[2025-05-07 20:01:38,861][INFO] End of Epoch 299 LR: 0.000000
[2025-05-07 20:01:40,729][INFO] Training Finished!
[2025-05-07 20:01:40,729][INFO] Best Test Accuracy (Top-1): 63.700% for 100 classes.
[2025-05-07 20:01:40,729][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c100e0.log


### c100e10
Epoch: 329
[2025-05-07 20:37:37,595][INFO] Epoch: [329][  0/127]	Time  1.998 ( 1.998)	Data  1.039 ( 1.039)	Loss 1.5491e+00 (1.5491e+00)	Acc@1  92.09 ( 92.09)	Mem: 1.87G/14.78G
[2025-05-07 20:38:07,003][INFO] Epoch: [329][ 50/127]	Time  0.640 ( 0.616)	Data  0.000 ( 0.021)	Loss 1.6125e+00 (1.5728e+00)	Acc@1  91.41 ( 90.85)	Mem: 1.87G/14.78G
[2025-05-07 20:38:40,775][INFO] Epoch: [329][100/127]	Time  0.684 ( 0.645)	Data  0.000 ( 0.011)	Loss 1.5459e+00 (1.5764e+00)	Acc@1  90.14 ( 90.83)	Mem: 1.87G/14.78G
[2025-05-07 20:38:58,070][INFO] Epoch 329 Train Summary: Loss: 1.574 | Acc@1: 90.827%
[2025-05-07 20:38:59,539][INFO] Test: [0/5]	Time  1.465 ( 1.465)	Loss 1.6803e+00 (1.6803e+00)	Acc@1  75.00 ( 75.00)	Mem: 1.87G/14.78G
[2025-05-07 20:39:00,451][INFO] * Test Acc@1 63.360 Loss 2.187
[2025-05-07 20:39:00,451][INFO] End of Epoch 329 LR: 0.000000
[2025-05-07 20:39:02,343][INFO] Training Finished!
[2025-05-07 20:39:02,343][INFO] Best Test Accuracy (Top-1): 63.660% for 100 classes.
[2025-05-07 20:39:02,343][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c100e10.log


Epoch: 299
[2025-05-07 20:05:43,274][INFO] Epoch: [299][  0/127]	Time  1.601 ( 1.601)	Data  1.099 ( 1.099)	Loss 1.6315e+00 (1.6315e+00)	Acc@1  89.45 ( 89.45)	Mem: 1.87G/14.78G
[2025-05-07 20:06:00,583][INFO] Epoch: [299][ 50/127]	Time  0.435 ( 0.371)	Data  0.000 ( 0.022)	Loss 1.5474e+00 (1.5862e+00)	Acc@1  91.70 ( 90.58)	Mem: 1.87G/14.78G
[2025-05-07 20:06:22,422][INFO] Epoch: [299][100/127]	Time  0.447 ( 0.403)	Data  0.000 ( 0.011)	Loss 1.6392e+00 (1.5837e+00)	Acc@1  91.21 ( 90.61)	Mem: 1.87G/14.78G
[2025-05-07 20:06:33,590][INFO] Epoch 299 Train Summary: Loss: 1.583 | Acc@1: 90.562%
[2025-05-07 20:06:34,947][INFO] Test: [0/5]	Time  1.355 ( 1.355)	Loss 1.6842e+00 (1.6842e+00)	Acc@1  74.61 ( 74.61)	Mem: 1.87G/14.78G
[2025-05-07 20:06:35,584][INFO] * Test Acc@1 63.020 Loss 2.191
[2025-05-07 20:06:35,584][INFO] End of Epoch 299 LR: 0.000021
[2025-05-07 20:06:35,584][INFO] 


### c100e30
Epoch: 194
[2025-05-07 23:45:39,168][INFO] Epoch: [194][ 0/64]	Time  4.008 ( 4.008)	Data  2.176 ( 2.176)	Loss 2.5615e+00 (2.5615e+00)	Acc@1  77.88 ( 77.88)	Mem: 1.94G/27.83G
[2025-05-07 23:46:39,052][INFO] Epoch: [194][50/64]	Time  1.265 ( 1.253)	Data  0.000 ( 0.043)	Loss 2.5779e+00 (2.5858e+00)	Acc@1  77.44 ( 77.06)	Mem: 1.94G/27.83G
[2025-05-07 23:46:54,382][INFO] Epoch 194 Train Summary: Loss: 2.588 | Acc@1: 77.028%
[2025-05-07 23:46:56,927][INFO] Test: [0/3]	Time  2.543 ( 2.543)	Loss 2.1964e+00 (2.1964e+00)	Acc@1  60.79 ( 60.79)	Mem: 1.95G/27.83G
[2025-05-07 23:46:57,562][INFO] * Test Acc@1 57.560 Loss 2.336
[2025-05-07 23:46:57,562][INFO] End of Epoch 194 LR: 0.000000
[2025-05-07 23:46:59,581][INFO] Training Finished!
[2025-05-07 23:46:59,582][INFO] Best Test Accuracy (Top-1): 57.580% for 100 classes.
[2025-05-07 23:46:59,582][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c100e30.log

Epoch: 149
[2025-05-07 22:43:51,908][INFO] Epoch: [149][ 0/64]	Time  3.852 ( 3.852)	Data  2.487 ( 2.487)	Loss 2.6380e+00 (2.6380e+00)	Acc@1  72.31 ( 72.31)	Mem: 1.94G/27.83G
[2025-05-07 22:44:51,725][INFO] Epoch: [149][50/64]	Time  1.220 ( 1.248)	Data  0.000 ( 0.049)	Loss 2.6601e+00 (2.6631e+00)	Acc@1  72.95 ( 73.65)	Mem: 1.94G/27.83G
[2025-05-07 22:45:05,011][INFO] Epoch 149 Train Summary: Loss: 2.661 | Acc@1: 73.647%
[2025-05-07 22:45:07,569][INFO] Test: [0/3]	Time  2.555 ( 2.555)	Loss 2.1884e+00 (2.1884e+00)	Acc@1  59.62 ( 59.62)	Mem: 1.95G/27.83G
[2025-05-07 22:45:08,253][INFO] * Test Acc@1 55.940 Loss 2.343
[2025-05-07 22:45:08,253][INFO] End of Epoch 149 LR: 0.000132
[2025-05-07 22:45:08,253][INFO] 


### c100e50
Epoch: 274
[2025-05-08 01:43:46,641][INFO] Epoch: [274][ 0/64]	Time  3.244 ( 3.244)	Data  2.046 ( 2.046)	Loss 2.9453e+00 (2.9453e+00)	Acc@1  79.15 ( 79.15)	Mem: 1.94G/27.84G
[2025-05-08 01:44:26,899][INFO] Epoch: [274][50/64]	Time  0.828 ( 0.853)	Data  0.000 ( 0.041)	Loss 2.9787e+00 (2.9437e+00)	Acc@1  79.59 ( 77.96)	Mem: 1.94G/27.84G
[2025-05-08 01:44:36,848][INFO] Epoch 274 Train Summary: Loss: 2.937 | Acc@1: 77.958%
[2025-05-08 01:44:39,264][INFO] Test: [0/3]	Time  2.414 ( 2.414)	Loss 2.2517e+00 (2.2517e+00)	Acc@1  61.28 ( 61.28)	Mem: 1.95G/27.84G
[2025-05-08 01:44:39,703][INFO] * Test Acc@1 57.220 Loss 2.402
[2025-05-08 01:44:39,704][INFO] End of Epoch 274 LR: 0.000000
[2025-05-08 01:44:41,443][INFO] Training Finished!
[2025-05-08 01:44:41,443][INFO] Best Test Accuracy (Top-1): 57.280% for 100 classes.
[2025-05-08 01:44:41,443][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c100e50.log

Epoch: 149
[2025-05-07 23:46:31,351][INFO] Epoch: [149][ 0/64]	Time  3.867 ( 3.867)	Data  1.925 ( 1.925)	Loss 3.2362e+00 (3.2362e+00)	Acc@1  61.77 ( 61.77)	Mem: 1.94G/27.84G
[2025-05-07 23:47:20,156][INFO] Epoch: [149][50/64]	Time  0.794 ( 1.033)	Data  0.000 ( 0.038)	Loss 3.2332e+00 (3.2285e+00)	Acc@1  61.72 ( 62.66)	Mem: 1.94G/27.84G
[2025-05-07 23:47:30,141][INFO] Epoch 149 Train Summary: Loss: 3.225 | Acc@1: 62.724%
[2025-05-07 23:47:32,580][INFO] Test: [0/3]	Time  2.436 ( 2.436)	Loss 2.2763e+00 (2.2763e+00)	Acc@1  58.25 ( 58.25)	Mem: 1.95G/27.84G
[2025-05-07 23:47:33,060][INFO] * Test Acc@1 54.880 Loss 2.419
[2025-05-07 23:47:33,061][INFO] End of Epoch 149 LR: 0.000441
[2025-05-07 23:47:33,061][INFO] 

### c100e100

Epoch: 149
[2025-05-07 13:32:18,197][INFO] Epoch: [149][ 0/32]	Time  7.015 ( 7.015)	Data  3.714 ( 3.714)	Loss 3.9606e+00 (3.9606e+00)	Acc@1  42.68 ( 42.68)	Mem: 2.09G/53.96G
[2025-05-07 13:33:29,766][INFO] Epoch 149 Train Summary: Loss: 3.971 | Acc@1: 43.064%
[2025-05-07 13:33:34,404][INFO] Test: [0/2]	Time  4.636 ( 4.636)	Loss 2.9715e+00 (2.9715e+00)	Acc@1  42.02 ( 42.02)	Mem: 2.10G/53.96G
[2025-05-07 13:33:34,695][INFO] * Test Acc@1 45.920 Loss 2.857
[2025-05-07 13:33:34,695][INFO] Saving new best checkpoint with Acc@1: 45.920%
[2025-05-07 13:33:36,040][INFO] End of Epoch 149 LR: 0.000000
[2025-05-07 13:33:36,969][INFO] Training Finished!
[2025-05-07 13:33:36,969][INFO] Best Test Accuracy (Top-1): 45.920% for 100 classes.
[2025-05-07 13:33:36,969][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c100e100.log

Epoch: 74
[2025-05-07 12:10:05,336][INFO] Epoch: [74][ 0/32]	Time  4.574 ( 4.574)	Data  3.832 ( 3.832)	Loss 4.1639e+00 (4.1639e+00)	Acc@1  30.49 ( 30.49)	Mem: 2.09G/53.96G
[2025-05-07 12:10:28,333][INFO] Epoch 74 Train Summary: Loss: 4.177 | Acc@1: 29.378%
[2025-05-07 12:10:32,333][INFO] Test: [0/2]	Time  3.998 ( 3.998)	Loss 3.3330e+00 (3.3330e+00)	Acc@1  32.67 ( 32.67)	Mem: 2.10G/53.96G
[2025-05-07 12:10:32,467][INFO] * Test Acc@1 36.240 Loss 3.226
[2025-05-07 12:10:32,467][INFO] Saving new best checkpoint with Acc@1: 36.240%
[2025-05-07 12:10:33,898][INFO] End of Epoch 74 LR: 0.000524
[2025-05-07 12:10:33,898][INFO] 


## 10 classes
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53601 train_ddp_flexible_log.py --data-path data_10_classes --num-classes 10 --log-file-base c10e0 --epochs 300 --checkpoint-name ckpt_10_classes_0p.pth --wrong-rate 0
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53602 train_ddp_flexible_log.py --data-path data_10_classes --num-classes 10 --log-file-base c10e10 --epochs 330 --checkpoint-name ckpt_10_classes_10p.pth --wrong-rate 0.09090909
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53603 train_ddp_flexible_log.py --data-path data_10_classes --num-classes 10 --log-file-base c10e30 --epochs 390 --checkpoint-name ckpt_10_classes_30p.pth --wrong-rate 0.23076923
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53604 train_ddp_flexible_log.py --data-path data_10_classes --num-classes 10 --log-file-base c10e50 --epochs 450 --checkpoint-name ckpt_10_classes_50p.pth --wrong-rate 0.33333333
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:53605 train_ddp_flexible_log.py --data-path data_10_classes --num-classes 10 --log-file-base c10e100 --epochs 300 --batch-size-per-gpu 256 --checkpoint-name ckpt_10_classes_100p.pth --wrong-rate 0.5


### c10e0
Epoch: 299
[2025-05-07 08:07:01,312][INFO] Epoch: [299][ 0/13]	Time  2.454 ( 2.454)	Data  1.183 ( 1.183)	Loss 1.4521e+00 (1.4521e+00)	Acc@1  57.13 ( 57.13)	Mem: 1.87G/14.78G
[2025-05-07 08:07:13,574][INFO] Epoch 299 Train Summary: Loss: 1.434 | Acc@1: 56.354%
[2025-05-07 08:07:14,731][INFO] Test: [0/1]	Time  1.155 ( 1.155)	Loss 1.3263e+00 (1.3263e+00)	Acc@1  59.92 ( 59.92)	Mem: 1.83G/14.78G
[2025-05-07 08:07:14,798][INFO] * Test Acc@1 59.921 Loss 1.326
[2025-05-07 08:07:14,798][INFO] End of Epoch 299 LR: 0.000000
[2025-05-07 08:07:17,640][INFO] Training Finished!
[2025-05-07 08:07:17,641][INFO] Best Test Accuracy (Top-1): 62.302% for 10 classes.
[2025-05-07 08:07:17,641][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c10e0.log

### c10e10

Epoch: 329
[2025-05-07 08:13:51,681][INFO] Epoch: [329][ 0/13]	Time  2.169 ( 2.169)	Data  1.140 ( 1.140)	Loss 1.6283e+00 (1.6283e+00)	Acc@1  55.76 ( 55.76)	Mem: 1.87G/14.78G
[2025-05-07 08:14:01,525][INFO] Epoch 329 Train Summary: Loss: 1.613 | Acc@1: 56.277%
[2025-05-07 08:14:02,431][INFO] Test: [0/1]	Time  0.904 ( 0.904)	Loss 1.3308e+00 (1.3308e+00)	Acc@1  60.71 ( 60.71)	Mem: 1.83G/14.78G
[2025-05-07 08:14:02,474][INFO] * Test Acc@1 60.714 Loss 1.331
[2025-05-07 08:14:02,474][INFO] End of Epoch 329 LR: 0.000000
[2025-05-07 08:14:04,926][INFO] Training Finished!
[2025-05-07 08:14:04,927][INFO] Best Test Accuracy (Top-1): 62.500% for 10 classes.
[2025-05-07 08:14:04,927][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c10e10.log

Epoch: 299
[2025-05-07 08:07:27,049][INFO] Epoch: [299][ 0/13]	Time  1.976 ( 1.976)	Data  1.081 ( 1.081)	Loss 1.6133e+00 (1.6133e+00)	Acc@1  54.30 ( 54.30)	Mem: 1.87G/14.78G
[2025-05-07 08:07:36,821][INFO] Epoch 299 Train Summary: Loss: 1.618 | Acc@1: 55.792%
[2025-05-07 08:07:37,773][INFO] Test: [0/1]	Time  0.950 ( 0.950)	Loss 1.3293e+00 (1.3293e+00)	Acc@1  61.51 ( 61.51)	Mem: 1.83G/14.78G
[2025-05-07 08:07:37,823][INFO] * Test Acc@1 61.508 Loss 1.329
[2025-05-07 08:07:37,823][INFO] End of Epoch 299 LR: 0.000021
[2025-05-07 08:07:37,823][INFO] 

### c10e30

Epoch: 389
[2025-05-07 08:24:20,071][INFO] Epoch: [389][ 0/13]	Time  2.093 ( 2.093)	Data  1.216 ( 1.216)	Loss 1.8877e+00 (1.8877e+00)	Acc@1  52.73 ( 52.73)	Mem: 1.87G/14.78G
[2025-05-07 08:24:27,503][INFO] Epoch 389 Train Summary: Loss: 1.899 | Acc@1: 50.892%
[2025-05-07 08:24:28,456][INFO] Test: [0/1]	Time  0.951 ( 0.951)	Loss 1.4775e+00 (1.4775e+00)	Acc@1  56.94 ( 56.94)	Mem: 1.83G/14.78G
[2025-05-07 08:24:28,516][INFO] * Test Acc@1 56.944 Loss 1.477
[2025-05-07 08:24:28,516][INFO] End of Epoch 389 LR: 0.000000
[2025-05-07 08:24:30,678][INFO] Training Finished!
[2025-05-07 08:24:30,678][INFO] Best Test Accuracy (Top-1): 58.929% for 10 classes.
[2025-05-07 08:24:30,678][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c10e30.log

Epoch: 299
[2025-05-07 08:07:59,748][INFO] Epoch: [299][ 0/13]	Time  2.102 ( 2.102)	Data  1.064 ( 1.064)	Loss 1.8601e+00 (1.8601e+00)	Acc@1  48.54 ( 48.54)	Mem: 1.87G/14.78G
[2025-05-07 08:08:09,453][INFO] Epoch 299 Train Summary: Loss: 1.907 | Acc@1: 47.600%
[2025-05-07 08:08:10,323][INFO] Test: [0/1]	Time  0.868 ( 0.868)	Loss 1.5192e+00 (1.5192e+00)	Acc@1  55.36 ( 55.36)	Mem: 1.83G/14.78G
[2025-05-07 08:08:10,374][INFO] * Test Acc@1 55.357 Loss 1.519
[2025-05-07 08:08:10,375][INFO] End of Epoch 299 LR: 0.000127
[2025-05-07 08:08:10,375][INFO]


### c10e50


Epoch: 449
[2025-05-07 08:32:09,965][INFO] Epoch: [449][ 0/13]	Time  1.476 ( 1.476)	Data  1.113 ( 1.113)	Loss 2.0372e+00 (2.0372e+00)	Acc@1  43.95 ( 43.95)	Mem: 1.87G/14.78G
[2025-05-07 08:32:13,190][INFO] Epoch 449 Train Summary: Loss: 2.053 | Acc@1: 43.885%
[2025-05-07 08:32:14,074][INFO] Test: [0/1]	Time  0.882 ( 0.882)	Loss 1.6682e+00 (1.6682e+00)	Acc@1  54.37 ( 54.37)	Mem: 1.83G/14.78G
[2025-05-07 08:32:14,127][INFO] * Test Acc@1 54.365 Loss 1.668
[2025-05-07 08:32:14,127][INFO] End of Epoch 449 LR: 0.000000
[2025-05-07 08:32:16,384][INFO] Training Finished!
[2025-05-07 08:32:16,384][INFO] Best Test Accuracy (Top-1): 54.563% for 10 classes.
[2025-05-07 08:32:16,384][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c10e50.log

Epoch: 299
[2025-05-07 08:08:02,885][INFO] Epoch: [299][ 0/13]	Time  2.068 ( 2.068)	Data  1.115 ( 1.115)	Loss 2.1002e+00 (2.1002e+00)	Acc@1  38.57 ( 38.57)	Mem: 1.87G/14.78G
[2025-05-07 08:08:12,849][INFO] Epoch 299 Train Summary: Loss: 2.098 | Acc@1: 39.031%
[2025-05-07 08:08:13,794][INFO] Test: [0/1]	Time  0.943 ( 0.943)	Loss 1.7386e+00 (1.7386e+00)	Acc@1  45.44 ( 45.44)	Mem: 1.83G/14.78G
[2025-05-07 08:08:13,854][INFO] * Test Acc@1 45.437 Loss 1.739
[2025-05-07 08:08:13,854][INFO] End of Epoch 299 LR: 0.000251
[2025-05-07 08:08:13,854][INFO] 


### c10e100

Epoch: 299
[2025-05-07 15:53:31,585][INFO] Epoch: [299][0/7]	Time  3.403 ( 3.403)	Data  2.046 ( 2.046)	Loss 2.1790e+00 (2.1790e+00)	Acc@1  41.11 ( 41.11)	Mem: 1.94G/27.83G
[2025-05-07 15:53:38,512][INFO] Epoch 299 Train Summary: Loss: 2.183 | Acc@1: 42.231%
[2025-05-07 15:53:39,433][INFO] Test: [0/1]	Time  0.919 ( 0.919)	Loss 1.8084e+00 (1.8084e+00)	Acc@1  49.40 ( 49.40)	Mem: 1.83G/27.83G
[2025-05-07 15:53:39,498][INFO] * Test Acc@1 49.405 Loss 1.808
[2025-05-07 15:53:39,499][INFO] End of Epoch 299 LR: 0.000000
[2025-05-07 15:53:41,614][INFO] Training Finished!
[2025-05-07 15:53:41,615][INFO] Best Test Accuracy (Top-1): 50.794% for 10 classes.
[2025-05-07 15:53:41,615][INFO] Checkpoints saved in ./checkpoint/. Log file: ./logs/c10e100.log
[2025-05-10 19:11:50,144][INFO] File logging to ./logs/c10e100.log initialized.

Epoch: 149
[2025-05-07 15:25:35,746][INFO] Epoch: [149][0/7]	Time  4.238 ( 4.238)	Data  2.636 ( 2.636)	Loss 2.2104e+00 (2.2104e+00)	Acc@1  37.11 ( 37.11)	Mem: 1.94G/27.83G
[2025-05-07 15:25:42,582][INFO] Epoch 149 Train Summary: Loss: 2.215 | Acc@1: 36.077%
[2025-05-07 15:25:43,492][INFO] Test: [0/1]	Time  0.908 ( 0.908)	Loss 1.9164e+00 (1.9164e+00)	Acc@1  43.65 ( 43.65)	Mem: 1.83G/27.83G
[2025-05-07 15:25:43,540][INFO] * Test Acc@1 43.651 Loss 1.916
[2025-05-07 15:25:43,541][INFO] End of Epoch 149 LR: 0.000499
[2025-05-07 15:25:43,541][INFO] 