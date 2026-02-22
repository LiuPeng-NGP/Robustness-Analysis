cd ImageClassification/CIFAR10_Classification_Structural
conda activate data_quality
# # Experiments results for CIFAR 10 with consistent mislabelled data
# No wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:52650 train_ddp_structural_mislabelled.py
Epoch: 199
 [=========================== 49/49 =============================>.]  Step: 42ms | Tot: 1s913ms | Loss: 0.001 | Acc: 100.000% (6250/6250)                       
 [=========================== 13/13 ==========================>....]  Step: 8ms | Tot: 143ms | Loss: 0.196 | Acc: 94.000% (1175/1250)                           
Test Loss: 3.047 | Test Acc: 94.170% (9417/10000)

# 10% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:50121 train_ddp_structural_mislabelled.py --wrong-rate 0.09090909 --checkpoint-name ckpt10p.pth Epoch: 219
 [=========================== 49/49 =============================>.]  Step: 118ms | Tot: 6s236ms | Loss: 0.312 | Acc: 100.000% (6250/6250)                      
 [=========================== 13/13 ==========================>....]  Step: 34ms | Tot: 446ms | Loss: 0.285 | Acc: 95.840% (1198/1250)                          
Test Loss: 4.255 | Test Acc: 94.150% (9415/10000)

# 30% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:50122 train_ddp_structural_mislabelled.py --wrong-rate 0.23076923 --checkpoint-name ckpt30p.pth --epochs 260                                                          
Epoch: 259
 [=========================== 49/49 =============================>.]  Step: 67ms | Tot: 3s705ms | Loss: 0.552 | Acc: 99.504% (6219/6250)                        
 [=========================== 13/13 ==========================>....]  Step: 9ms | Tot: 264ms | Loss: 0.558 | Acc: 92.160% (1152/1250)                           
Test Loss: 7.688 | Test Acc: 91.280% (9128/10000)

# 50% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:50123 train_ddp_structural_mislabelled.py --wrong-rate 0.33333333 --checkpoint-name ckpt50p.pth --epochs 300
Epoch: 299
 [=========================== 49/49 =============================>.]  Step: 46ms | Tot: 2s955ms | Loss: 0.704 | Acc: 97.264% (6079/6250)                        
 [=========================== 13/13 ==========================>....]  Step: 10ms | Tot: 205ms | Loss: 0.783 | Acc: 88.240% (1103/1250)                          
Test Loss: 10.200 | Test Acc: 87.990% (8799/10000)

# 100% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:50124 train_ddp_structural_mislabelled.py --wrong-rate 0.5 --checkpoint-name ckpt100p.pth --epochs 400
 [=========================== 49/49 =============================>.]  Step: 34ms | Tot: 1s901ms | Loss: 0.763 | Acc: 54.832% (3427/6250)                        
 [=========================== 13/13 ==========================>....]  Step: 8ms | Tot: 165ms | Loss: 1.059 | Acc: 39.840% (498/1250)                            
Test Loss: 13.941 | Test Acc: 40.720% (4072/10000)