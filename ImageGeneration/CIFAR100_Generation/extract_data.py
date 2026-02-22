import torch
from torchvision.datasets import CIFAR100
import os

train_set = CIFAR100("./data", train=True, download=True)
print("CIFAR100 train dataset:", len(train_set))

images = []
labels = []
for img, label in train_set:
    images.append(img)
    labels.append(label)

labels = torch.tensor(labels)
for i in range(10):
    assert (labels == i).sum() == 500

output_dir = "./data/cifar100-pngs/"
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
for i, pil in enumerate(images):
    pil.save(os.path.join(output_dir, "{:05d}.png".format(i)))