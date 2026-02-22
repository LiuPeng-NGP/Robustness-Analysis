cd ImageGeneration/CIFAR10_Generation
conda activate data_quality
# No wrong data
## Train
1. torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51653 train.py --config config/edm_conditional.yaml --use_amp

## Sample
1. torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51655 sample.py --config config/edm_conditional.yaml --use_amp

## Test Accuracy
python3 ImageClassification/CIFAR10_Classification/accuracy_gen_images.py --folder_path ImageGeneration/CIFAR10_Generation/results/1200_conditional_EDM/EMAgenerated_ep1199_edm_steps18/pngs --checkpoint_path ImageClassification/CIFAR10_Classification/checkpoint/ckptddp.pth


## Record
cd ImageGeneration/CIFAR10_Generation
python3 extract_data.py
python3 -m pytorch_fid results/1200_conditional_EDM/EMAgenerated_ep1199_edm_steps18/pngs data/cifar10-pngs
python3 -m pytorch_fid results/1200_conditional_EDM_10_percent/EMAgenerated_ep1319_edm_steps18/pngs data/cifar10-pngs
python3 -m pytorch_fid results/1200_conditional_EDM_30_percent/EMAgenerated_ep1559_edm_steps18/pngs data/cifar10-pngs
python3 -m pytorch_fid results/1200_conditional_EDM_50_percent/EMAgenerated_ep1799_edm_steps18/pngs data/cifar10-pngs
python3 -m pytorch_fid results/1200_conditional_EDM_100_percent/EMAgenerated_ep2399_edm_steps18/pngs data/cifar10-pngs
1. c10e0    FID:  3.4882287187339784
2. c10e10   FID:  3.6802004245327566
3. c10e30   FID:  3.6588375784389770
4. c10e50   FID:  3.6551971440234183
5. c10e100  FID:  3.621641330167506


## Wrong data record
1. no wrong data 94.082%
2. 10% wrong data 84.16%
3. 30% wrong data 69.864%
4. 50% wrong data 57.876%
5. 100% wrong data 40.630%

# 10_percent
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51651 train.py --config config/edm_conditional_10_percent.yaml --use_amp
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51652 sample.py --config config/edm_conditional_10_percent.yaml --use_amp

python3 ImageClassification/CIFAR10_Classification/accuracy_gen_images.py --folder_path ImageGeneration/CIFAR10_Generation/results/1200_conditional_EDM_10_percent/EMAgenerated_ep1319_edm_steps18/pngs --checkpoint_path ImageClassification/CIFAR10_Classification/checkpoint/ckptddp.pth

# 30_percent
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51661 train.py --config config/edm_conditional_30_percent.yaml --use_amp
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51662 sample.py --config config/edm_conditional_30_percent.yaml --use_amp

python3 ImageClassification/CIFAR10_Classification/accuracy_gen_images.py --folder_path ImageGeneration/CIFAR10_Generation/results/1200_conditional_EDM_30_percent/EMAgenerated_ep1559_edm_steps18/pngs --checkpoint_path ImageClassification/CIFAR10_Classification/checkpoint/ckptddp.pth


# 50_percent
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51671 train.py --config config/edm_conditional_50_percent.yaml --use_amp
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51672 sample.py --config config/edm_conditional_50_percent.yaml --use_amp

python3 ImageClassification/CIFAR10_Classification/accuracy_gen_images.py --folder_path ImageGeneration/CIFAR10_Generation/results/1200_conditional_EDM_50_percent/EMAgenerated_ep1799_edm_steps18/pngs --checkpoint_path ImageClassification/CIFAR10_Classification/checkpoint/ckptddp.pth


# 100_percent
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51681 train.py --config config/edm_conditional_100_percent.yaml --use_amp
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51682 sample.py --config config/edm_conditional_100_percent.yaml --use_amp

python3 ImageClassification/CIFAR10_Classification/accuracy_gen_images.py --folder_path ImageGeneration/CIFAR10_Generation/results/1200_conditional_EDM_100_percent/EMAgenerated_ep2399_edm_steps18/pngs --checkpoint_path ImageClassification/CIFAR10_Classification/checkpoint/ckptddp.pth