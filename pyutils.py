import numpy as np
import os
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def population_mean_norm(path):
    train_dataset1 = torchvision.datasets.ImageFolder(
            root=path,

            transform=transforms.Compose([
                    transforms.ToTensor()
                    ])

        )

    dataloader = torch.utils.data.DataLoader(train_dataset1, batch_size=4096, shuffle=False, num_workers=4)

    print("starting normalization")

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for data,label in dataloader:
        # shape (batch_size, 3, height, width)
        numpy_image = data.numpy()
        
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)

    return pop_mean, pop_std0


def show(img, title, epoch, orig):
    npimg = img.numpy()
    plt.figure()
    plt.title(title)
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig('results/results'+str(orig)+str(epoch)+'.png')

