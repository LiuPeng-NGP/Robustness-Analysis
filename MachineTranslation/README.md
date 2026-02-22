# Experiment Design Use this
1. Transformer Architecture.
2. WMT 2014 as dataset
3. eraly checkpoint during training for low qualtiy data generation
4. BLEU for evaluation

# Experiment Design 2
1. gpt 2 architecture
2. the new dataset WMT 2024
3. NLL and BLEU for evaluation

# Environment
cd MachineTranslation
conda activate data_quality_mtts

# Prepare data
python3 prepare_data.py
python3 get_subset.py
(After this, we get Number of new training examples: 287113)
python3 bpe.py

# Train to early checkpoints and thus low quality data
torchrun --nproc_per_node=8 train_correct.py
torchrun --nproc_per_node=8 create_noisy_dataset.py

# Training
torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:50121 train_low_quality.py --noise_ratio 0.09090909 --total_steps 50000
torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:50122 train_low_quality.py --noise_ratio 0.23076923 --total_steps 50000
torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:50123 train_low_quality.py --noise_ratio 0.33333333 --total_steps 50000
torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:50124 train_low_quality.py --noise_ratio 0.5 --total_steps 50000
