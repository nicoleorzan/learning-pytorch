from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import random_split

from agents import Sender
from agents import Receiver
import numpy as np
import torch
from torch.optim import Adam

# load dataset

dataset = MNIST(root="../data", transform=ToTensor())

validation_size = 10000
batch_size = 80
half_batch_size = int(batch_size/2)
train_size = len(dataset) - validation_size

train_ds, val_ds = random_split(dataset, [train_size, validation_size])

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

# define agents
input_size = 28*28
hidden_size = 32
n_actions = 2
vocab_size = 10

# train hyperparams
num_episodes = 500
lr = 0.1

send = Sender(in_size=input_size, hidden_size=hidden_size, 
        vocab_len=vocab_size, lr=lr)

recv = Receiver(in_size=input_size, vocabulary_size_sender = vocab_size, 
    hidden_size=hidden_size, n_actions = n_actions, lr=lr)


def get_images(imgs, labels):
    
    num_images = int(len(labels)/2)
    imgsa = imgs[:num_images]
    imgsb = imgs[num_images:]
    imgsa_receiver = torch.empty(imgsa.shape)
    imgsb_receiver = torch.empty(imgsa.shape)

    targets = np.random.randint(2, size=num_images)
    
    for i in range(int(len(labels)/2)):
        if targets[i]==0:
            imgsa_receiver[i] = imgsa[i]
            imgsb_receiver[i] = imgsb[i]
        else:
            imgsa_receiver[i] = imgsb[i]
            imgsb_receiver[i] = imgsa[i]

    return imgsa, imgsb, imgsa_receiver, imgsb_receiver, targets, targets

def get_reward(act, imgsa_s, imgsa_r, imgsb_r):
    rewards = torch.empty([len(act)])

    for idx, val in enumerate(act.numpy()):
        rewards[idx] = 0
        if (val == 0):
            if (torch.equal(imgsa_r[idx], imgsa_s[idx]) ):
                rewards[idx] = 1
        if (val == 1):
            if (torch.equal(imgsb_r[idx], imgsa_s[idx]) ):
                rewards[idx] = 1              
    return rewards

#TRAIN LOOP

send_opt = Adam(send.model.parameters(), lr=lr)
recv_opt = Adam(recv.model.parameters(), lr=lr)

for ep in range(num_episodes):
    print("episode=", ep)

    losslist_r = []; losslist_s = []; acclist_s = [];  acclist_r = []

    for imgs, labs in train_loader:

        imgsa_s, imgsb_s, imgsa_r, imgsb_r, send_targets, recv_targets = get_images(imgs, labs)

        mex, logprobs_s = send.model(imgsa_s, imgsb_s)
        #print("mex=", mex)
        act, logprobs_r = recv.model(imgsa_r, imgsb_r, mex.detach())
        #print("act=", act)
        #print("logprobs_r=", logprobs_r)

        # COMPUTE ERROR 
        send_error = torch.abs(act - torch.tensor(send_targets)) 
        #print("error=", send_error)
        #print("CHECK REQUIRES GRAD")
        #print("act", act.requires_grad)
        #print("logprobs", logprobs_s.requires_grad)
        #print("send_targets", torch.tensor(send_targets).requires_grad)
        #print("send_error", send_error.requires_grad)
        # SENDER LOSS
        send_loss, send_acc = send.loss(send_error, logprobs_s)

        send_opt.zero_grad()
        send_loss.backward()
        send_opt.step()

        
        # RECEIVER LOSS
        recv_error = torch.abs(act.detach() - recv_targets) 
        recv_loss, recv_acc = recv.loss(recv_error, logprobs_r)

        recv_opt.zero_grad()
        recv_loss.backward()
        recv_opt.step()

        losslist_s.append(send_loss.detach().numpy())
        losslist_r.append(recv_loss.detach().numpy())        
        acclist_s.append(send_acc.detach().numpy())
        acclist_r.append(recv_acc.detach().numpy())
       
    print("sender loss", np.mean(losslist_s))
    print("receiver loss", np.mean(losslist_r))
    
    print("sender accuracy", np.mean(acclist_s))
    print("receiver accuracy", np.mean(acclist_r))


torch.save(send.model.state_dict(), 'sender_model_mnist.pth')
torch.save(recv.model.state_dict(), 'receiver_model_mnist.pth')