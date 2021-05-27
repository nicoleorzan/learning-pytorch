import torch
from torch.optim import Adam
import numpy as np

from agents import Sender, Receiver
from utils import get_images, show_batch, get_data
from losses import accuracy, reward

# load dataset
def main():
    train_loader, val_loader, input_size = get_data(100)

    show_batch(train_loader)

    # define agents
    hidden_size = 32
    n_actions = 2
    vocab_size = 11

    # train hyperparams
    num_episodes = 100
    lr = 0.1

    send = Sender(in_size=input_size, hidden_size=hidden_size, 
            vocab_len=vocab_size, lr=lr)

    recv = Receiver(in_size=input_size, vocabulary_size_sender = vocab_size, 
        hidden_size=hidden_size, n_actions = n_actions, lr=lr)


    def batch(send, recv, images_batch, labels_batch, send_opt=None, recv_opt=None):

        imgsa_s, imgsb_s, imgsa_r, imgsb_r, targets, _ = get_images(images_batch, labels_batch)

        probs_s, message, logprobs_s, entropy_s = send.model(imgsa_s, imgsb_s)
        probs_r, actions, logprobs_r, entropy_r = recv.model(imgsa_r, imgsb_r, message.detach())

        error = reward(actions, targets) #torch.abs(act - targets) il - e` gia nell'update mi pare
        acc = accuracy(actions, targets) #torch.mean(error.detach().double())

        send_loss = send.loss(error, logprobs_s, entropy_s)
        recv_loss = recv.loss(error, logprobs_r, entropy_r)

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

    # UPLOAD MODELS

    #send.model.load_state_dict(torch.load('sender_model_mnist.pth'))
    #recv.model.load_state_dict(torch.load('receiver_model_mnist.pth'))

    send_opt = Adam(send.model.parameters(), lr=lr)
    recv_opt = Adam(recv.model.parameters(), lr=lr)
    print("lr=", lr)

    #TRAIN LOOP

    train_send_losses, train_recv_losses, val_send_losses, val_recv_losses, val_accuracy = [], [], [], [], []

    for ep in range(num_episodes):
        print("episode=", ep)

        # TRAIN STEP
        print("train")
        for imgs, labs in train_loader:

            train_error, train_send_loss, train_recv_loss, _, train_acc = batch(send, recv, imgs, labs, send_opt, recv_opt)

        print("evaluation")
        # EVALUATION STEP
        with torch.no_grad():

            results = [ batch(send, recv, imgs, labs) for imgs, labs in val_loader ]
            
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

    torch.save(send.model.state_dict(), 'sender_model_cifar_01.pth')
    torch.save(recv.model.state_dict(), 'receiver_model_cifar_01.pth')


if __name__ == "__main__":
    main()