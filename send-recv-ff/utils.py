import torch
import numpy as np

def accuracy(outputs, labels):
    acc = torch.tensor(torch.sum(outputs == labels).item()/len(labels))
    return acc

def get_images(imgs):
    
    num_images = int(len(imgs)/2)
    imgsa = imgs[:num_images]
    imgsb = imgs[num_images:]
    imgsa_receiver = torch.empty(imgsa.shape)
    imgsb_receiver = torch.empty(imgsa.shape)

    targets = torch.tensor(np.random.randint(2, size=num_images))
    
    for i in range(num_images):
        if targets[i]==0:
            imgsa_receiver[i] = imgsa[i]
            imgsb_receiver[i] = imgsb[i]
        else:
            imgsa_receiver[i] = imgsb[i]
            imgsb_receiver[i] = imgsa[i]

    return imgsa, imgsb, imgsa_receiver, imgsb_receiver, targets