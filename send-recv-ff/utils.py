import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

import matplotlib.pyplot as plt 
import numpy as np

def get_data(batch_size, val_percentage=0.2):

    #data_dir = '../data/cifar10'
    #dataset = ImageFolder(data_dir + "/train", transform=ToTensor())
    dataset = MNIST(root="../data", transform=ToTensor())

    img, _ = dataset[0]
    input_size = np.prod(img.shape)

    print("len dataset=", len(dataset))

    validation_size = int(len(dataset)*val_percentage)
    train_size = len(dataset) - validation_size
    print("train_size=", train_size, ", validation_size=", validation_size)

    train_ds, val_ds = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

    return train_loader, val_loader, input_size

def get_images(imgs, labels):
    
    num_images = int(len(imgs)/2)
    imgsa = imgs[:num_images]
    imgsb = imgs[num_images:]
    imgsa_receiver = torch.empty(imgsa.shape)
    imgsb_receiver = torch.empty(imgsa.shape)
    labels_target = torch.tensor(np.random.randint(2, size=num_images))

    targets = torch.tensor(np.random.randint(2, size=num_images))
    
    for i in range(num_images):
        labels_target[i] = labels[i]
        if targets[i]==0:
            imgsa_receiver[i] = imgsa[i]
            imgsb_receiver[i] = imgsb[i]
        else:
            imgsa_receiver[i] = imgsb[i]
            imgsb_receiver[i] = imgsa[i]

    return imgsa, imgsb, imgsa_receiver, imgsb_receiver, targets, labels_target

def show_batch(dl):
    for imgs, labs in dl:
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(imgs, 10).permute(1,2,0))
        plt.show()
        break