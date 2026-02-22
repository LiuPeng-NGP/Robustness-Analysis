cd ImageClassification/CIFAR100_Classification_Structural
conda activate data_quality
# Experiments results for CIFAR 100 with consistent mislabelled data
# No wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51650 train_ddp.py                             
Epoch: 199
 [=========================== 49/49 =============================>.]  Step: 85ms | Tot: 4s323ms | Loss: 0.007 | Acc: 100.000% (6250/6250)                       
 [=========================== 13/13 ==========================>....]  Step: 14ms | Tot: 241ms | Loss: 0.880 | Acc: 78.000% (975/1250)                           
Test Loss: 13.166 | Test Acc: 75.540% (7554/10000)
# 10% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51121 train_ddp.py --wrong-rate 0.09090909 --checkpoint-name ckpt10p.pth --epochs 220                           
Epoch: 219
 [=========================== 49/49 =============================>.]  Step: 68ms | Tot: 3s362ms | Loss: 0.312 | Acc: 99.984% (6249/6250)                        
 [=========================== 13/13 ==========================>....]  Step: 4ms | Tot: 209ms | Loss: 1.195 | Acc: 74.800% (935/1250)                            
Test Loss: 15.208 | Test Acc: 75.830% (7583/10000)

# 30% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51122 train_ddp.py --wrong-rate 0.23076923 --checkpoint-name ckpt30p.pth --epochs 260                              
Epoch: 259
 [=========================== 49/49 =============================>.]  Step: 49ms | Tot: 2s401ms | Loss: 0.554 | Acc: 99.936% (6246/6250)                        
 [=========================== 13/13 ==========================>....]  Step: 8ms | Tot: 150ms | Loss: 1.379 | Acc: 75.200% (940/1250)                            
Test Loss: 18.255 | Test Acc: 74.050% (7405/10000)

# 50% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51123 train_ddp.py --wrong-rate 0.33333333 --checkpoint-name ckpt50p.pth --epochs 300                               
Epoch: 299
 [=========================== 49/49 =============================>.]  Step: 32ms | Tot: 1s384ms | Loss: 0.792 | Acc: 92.688% (5793/6250)                        
 [=========================== 13/13 ==========================>....]  Step: 3ms | Tot: 97ms | Loss: 2.001 | Acc: 62.560% (782/1250)                             
Test Loss: 26.200 | Test Acc: 62.440% (6244/10000)

# 100% wrong data
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:51124 train_ddp.py --wrong-rate 0.5 --checkpoint-name ckpt100p.pth --epochs 400                            
Epoch: 399
 [=========================== 49/49 =============================>.]  Step: 13ms | Tot: 756ms | Loss: 0.891 | Acc: 49.952% (3122/6250)                          
 [=========================== 13/13 ==========================>....]  Step: 3ms | Tot: 60ms | Loss: 2.231 | Acc: 32.400% (405/1250)                             
Test Loss: 29.703 | Test Acc: 33.390% (3339/10000)