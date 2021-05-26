from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import random_split

from agents import Sender
from agents import Receiver
import numpy as np
import torch
from torch.optim import Adam

from utils import get_images, accuracy

# load dataset

dataset = MNIST(root="../data", transform=ToTensor())

validation_size = 10000
batch_size = 100
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
num_episodes = 200
lr = 0.1

send = Sender(in_size=input_size, hidden_size=hidden_size, 
        vocab_len=vocab_size, lr=lr)

recv = Receiver(in_size=input_size, vocabulary_size_sender = vocab_size, 
    hidden_size=hidden_size, n_actions = n_actions, lr=lr)

def reward(act, targets):

    rew = torch.zeros(act.shape)

    for i in range(len(act)):
        if (act[i] == targets[i]):
            rew[i] = 1
            
    return rew


def batch(send, recv, images_batch, send_opt=None, recv_opt=None):

    imgsa_s, imgsb_s, imgsa_r, imgsb_r, targets = get_images(images_batch)

    mex, logprobs_s, entropy_s = send.model(imgsa_s, imgsb_s)
    act, logprobs_r, entropy_r = recv.model(imgsa_r, imgsb_r, mex.detach())

    error = reward(act, targets) #torch.abs(act - targets) il - e` gia nell'update mi pare
    acc = accuracy(act, targets) #torch.mean(error.detach().double())

    send_loss = send.loss(error, logprobs_s)
    recv_loss = recv.loss(error, logprobs_r)

    if send_opt is not None:

        # SENDER LOSS
        send_opt.zero_grad()
        send_loss.backward()
        send_opt.step()

    if recv_opt is not None:

        # RECEIVER LOSS
        recv_opt.zero_grad()
        recv_loss.backward()
        recv_opt.step()

    return error, send_loss, recv_loss, len(imgsa_s), acc


#TRAIN LOOP
#send.model.load_state_dict(torch.load('sender_model_mnist.pth'))
#recv.model.load_state_dict(torch.load('receiver_model_mnist.pth'))

send_opt = Adam(send.model.parameters(), lr=lr)
recv_opt = Adam(recv.model.parameters(), lr=lr)

train_send_losses, train_recv_losses, val_send_losses, val_recv_losses, val_accuracy = [], [], [], [], []

for ep in range(num_episodes):
    print("episode=", ep)

    # TRAIN STEP
    print("train")
    for imgs, _ in train_loader:

        train_error, train_send_loss, train_recv_loss, _, train_acc = batch(send, recv, imgs, send_opt, recv_opt)

    print("evaluation")
    # EVALUATION STEP
    with torch.no_grad():

        results = [ batch(send, recv, imgs) for imgs, _ in val_loader ]
        
    val_error, val_send_loss, val_recv_loss, nums, val_acc = zip(*results)

    total = np.sum(nums)
    send_train_avg_loss = np.sum(np.multiply(train_send_loss.detach().numpy(), nums))/total
    recv_train_avg_loss = np.sum(np.multiply(train_recv_loss.detach().numpy(), nums))/total
    train_send_losses.append(send_train_avg_loss)
    train_recv_losses.append(recv_train_avg_loss)

    send_val_avg_loss = np.sum(np.multiply(val_send_loss, nums))/total
    recv_val_avg_loss = np.sum(np.multiply(val_recv_loss, nums))/total
    val_send_losses.append(send_val_avg_loss)
    val_recv_losses.append(recv_val_avg_loss)
        
    val_avg_accuracy = np.sum(np.multiply(val_acc, nums))/total
    val_accuracy.append(val_avg_accuracy)

    print("sender train loss", send_train_avg_loss)
    print("receiver train loss", recv_train_avg_loss)
    print("sender val loss", send_val_avg_loss)
    print("receiver val loss", recv_val_avg_loss)
    print("accuracy", val_avg_accuracy)
    print("\n")


torch.save(send.model.state_dict(), 'sender_model_mnist.pth')
torch.save(recv.model.state_dict(), 'receiver_model_mnist.pth')