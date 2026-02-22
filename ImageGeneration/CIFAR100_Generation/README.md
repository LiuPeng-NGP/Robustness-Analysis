cd ImageGeneration/CIFAR100_Generation
# No wrong data
## Train
1. torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:52653 train.py --config config/edm_conditional.yaml --use_amp

## Sample
1. torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:52655 sample.py --config config/edm_conditional.yaml --use_amp

## Test Accuracy
python3 ImageClassification/CIFAR100_Classification/accuracy_gen_images.py --folder_path ImageGeneration/CIFAR100_Generation/results/1200_conditional_EDM/EMAgenerated_ep1199_edm_steps18/pngs --checkpoint_path ImageClassification/CIFAR100_Classification/checkpoint/ckptddp.pth

## Record
cd ImageGeneration/CIFAR100_Generation
python3 extract_data.py
python3 -m pytorch_fid results/1200_conditional_EDM/EMAgenerated_ep1199_edm_steps18/pngs data/cifar100-pngs
python3 -m pytorch_fid results/1200_conditional_EDM_10_percent/EMAgenerated_ep1319_edm_steps18/pngs data/cifar100-pngs
python3 -m pytorch_fid results/1200_conditional_EDM_30_percent/EMAgenerated_ep1559_edm_steps18/pngs data/cifar100-pngs
python3 -m pytorch_fid results/1200_conditional_EDM_50_percent/EMAgenerated_ep1799_edm_steps18/pngs data/cifar100-pngs
python3 -m pytorch_fid results/1200_conditional_EDM_100_percent/EMAgenerated_ep2399_edm_steps18/pngs data/cifar100-pngs
1. c100e0   FID:  5.380963731568954
2. c100e10  FID:  5.705028480720614
3. c100e30  FID:  6.096602464465832
4. c100e50  FID:  6.122614765241906
5. c100e100 FID:  6.284733275444012

## CIFAR 100 classification and generation
1. classification ddp amp 77.11%  single 78.96
2. origin   65.236%
3. 10%      54.262%
4. 30%      38.882%
5. 50%      30.404%
6. 100%     16.996%


# 10_percent                    
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:52651 train.py --config config/edm_conditional_10_percent.yaml --use_amp
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:52652 sample.py --config config/edm_conditional_10_percent.yaml --use_amp

python3 ImageClassification/CIFAR100_Classification/accuracy_gen_images.py --folder_path ImageGeneration/CIFAR100_Generation/results/1200_conditional_EDM_10_percent/EMAgenerated_ep1319_edm_steps18/pngs --checkpoint_path ImageClassification/CIFAR100_Classification/checkpoint/ckptddp.pth


# 30_percent                   
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:52661 train.py --config config/edm_conditional_30_percent.yaml --use_amp
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:52662 sample.py --config config/edm_conditional_30_percent.yaml --use_amp

python3 ImageClassification/CIFAR100_Classification/accuracy_gen_images.py --folder_path ImageGeneration/CIFAR100_Generation/results/1200_conditional_EDM_30_percent/EMAgenerated_ep1559_edm_steps18/pngs --checkpoint_path ImageClassification/CIFAR100_Classification/checkpoint/ckptddp.pth


# 50_percent                    
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:52671 train.py --config config/edm_conditional_50_percent.yaml --use_amp
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:52672 sample.py --config config/edm_conditional_50_percent.yaml --use_amp

python3 ImageClassification/CIFAR100_Classification/accuracy_gen_images.py --folder_path ImageGeneration/CIFAR100_Generation/results/1200_conditional_EDM_50_percent/EMAgenerated_ep1799_edm_steps18/pngs --checkpoint_path ImageClassification/CIFAR100_Classification/checkpoint/ckptddp.pth


# 100_percent                   
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:52681 train.py --config config/edm_conditional_100_percent.yaml --use_amp
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:52682 sample.py --config config/edm_conditional_100_percent.yaml --use_amp

python3 ImageClassification/CIFAR100_Classification/accuracy_gen_images.py --folder_path ImageGeneration/CIFAR100_Generation/results/1200_conditional_EDM_100_percent/EMAgenerated_ep2399_edm_steps18/pngs --checkpoint_path ImageClassification/CIFAR100_Classification/checkpoint/ckptddp.pth