

cd TextGeneration
conda activate data_quality_text
# Dataset
cd data/openwebtext
python3 prepare.py



# Train
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_10_percent_wrong.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_30_percent_wrong_2x_batch.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_50_percent_wrong_2x_batch.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_100_percent_wrong_12x_batch.py

# Train no addition
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_10_percent_wrong_wo_add.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_30_percent_wrong_2x_batch_wo_add.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_50_percent_wrong_12x_batch_wo_add.py

# Train no addition same ratio as original
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_10_percent_wrong_wo_add_o.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_30_percent_wrong_2x_batch_wo_add_o.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_50_percent_wrong_2x_batch_wo_add_o.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_100_percent_wrong_12x_batch_wo_add_o.py

# test gradient
torchrun --standalone --nproc_per_node=8 --rdzv_endpoint=localhost:52661 test_loss_ddp.py --batch_size=12 --gradient_accumulation_steps=5
torchrun --standalone --nproc_per_node=8 --rdzv_endpoint=localhost:52662 test_loss_ddp.py --batch_size=12 --gradient_accumulation_steps=10
torchrun --standalone --nproc_per_node=8 --rdzv_endpoint=localhost:52663 test_loss_ddp.py --batch_size=12 --gradient_accumulation_steps=20
torchrun --standalone --nproc_per_node=8 --rdzv_endpoint=localhost:52664 test_loss_ddp.py --batch_size=12 --gradient_accumulation_steps=40
torchrun --standalone --nproc_per_node=8 --rdzv_endpoint=localhost:52665 test_loss_ddp.py --batch_size=12 --gradient_accumulation_steps=80

# test gradient directly
torchrun --standalone --nproc_per_node=8 test_grad_lb_ddp.py


# Reference
https://github.com/karpathy/nanoGPT