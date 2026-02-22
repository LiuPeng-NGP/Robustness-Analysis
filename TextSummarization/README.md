# Experiment Design
1. CNN/Daily Mail dataset
2. Transformer architecture
3. BLEU for text summarization

Number of training examples: 287113
# Prepare Dataset
cd TextSummarization
conda activate data_quality_mtts
python3 prepare_data.py
python3 bpe.py

# create low quality dataset
torchrun --nproc_per_node=8 train_correct.py
torchrun --nproc_per_node=8 create_noisy_dataset.py

# Train

torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:50121 train_noisy_summarization.py --noise_ratio 0.09090909 --total_steps 50000
torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:50122 train_noisy_summarization.py --noise_ratio 0.23076923 --total_steps 50000
torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:50123 train_noisy_summarization.py --noise_ratio 0.33333333 --total_steps 50000
torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:50124 train_noisy_summarization.py --noise_ratio 0.5 --total_steps 50000
