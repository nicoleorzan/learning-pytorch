import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class Sender_Net(nn.Module):

    def __init__(self, in_size, hidden_size, vocab_len, lr):
        
        super(Sender_Net, self).__init__()
        self.policy_single = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Sigmoid()
        )
        self.policy_combined = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, vocab_len),
            nn.LogSoftmax(dim=1)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, img1, img2):
        
        #print("img1.shape", img1.shape)

        img1 = img1.view(img1.size(0), -1)
        #print("img1.shape", img1.shape)
        img2 = img2.view(img2.size(0), -1)
    
        out1 = self.policy_single(img1)
        #print("out1.shape", out1.shape)

        out2 = self.policy_single(img2)

        combined = torch.cat((out1, out2),dim=1) # attaccate usando asse x (una sopra l'altra)
        probs = self.policy_combined(combined)

        # sampling
        dist = Categorical(probs=probs)
        
        if self.training:
            actions = dist.sample()
        else:
            actions = dist.argmax(dim=1)

        logprobs = dist.log_prob(actions)

        entropy = dist.entropy()
        
        return probs, actions, logprobs, entropy

class Sender():

    def __init__(self, in_size, hidden_size, vocab_len, lr = 0.5):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.vocab_len = vocab_len
        self.lr = lr

        self.model = Sender_Net(self.in_size, self.hidden_size, self.vocab_len, self.lr)

        self.training = True
        self.baseline = 0
        self.ent_reg = 0
        self.n = 0

    def loss(self, error, logprobs, entropy):

        # added regularization to decrease variance 

        #policy_loss = ((error.detach() - self.baseline)*(-logprobs)).mean()
        #entropy_loss = -entropy.mean() * self.ent_reg

        loss = (-logprobs * error.detach()).mean() #policy_loss + entropy_loss 

        if self.training:
            self.n += 1.
            #self.baseline += (error.detach().mean().item() - self.baseline) / self.n
        
        return loss


class Receiver_Net(nn.Module):

    def __init__(self, in_size, vocab_len, hidden_size, n_actions, lr):
        super(Receiver_Net, self).__init__()
                
        self.vocab_len = vocab_len 

        self.policy_single_img = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Sigmoid()
        )
        self.policy_single_mex = nn.Sequential(
            nn.Linear(self.vocab_len, hidden_size),
            nn.ReLU()
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.softmax = nn.Softmax(dim=1)
               
    def forward(self, img1, img2, mex):

        mex = torch.nn.functional.one_hot(mex, num_classes=self.vocab_len)

        img1 = img1.view(img1.size(0), -1)
        img2 = img2.view(img2.size(0), -1)
        out1 = self.policy_single_img(img1)
        out2 = self.policy_single_img(img2)

        symbol = mex.view(mex.size(0), -1).float()
        symbol = self.policy_single_mex(symbol)

        out1 = torch.bmm(symbol.view(symbol.size(0), 1, symbol.size(1)), out1.view(out1.size(0), symbol.size(1), 1)) # un numero per ogni immagine
        out2 = torch.bmm(symbol.view(symbol.size(0), 1, symbol.size(1)), out2.view(out2.size(0), symbol.size(1), 1)) 

        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)

        combined = torch.cat((out1, out2),dim=1)
        probs = self.softmax(combined)

        dist = Categorical(probs=probs)
        
        if self.training:
            actions = dist.sample()
        else:
            actions = dist.argmax(dim=1)

        logprobs = dist.log_prob(actions)

        entropy = dist.entropy()

        return probs, actions, logprobs, entropy

class Receiver():

    def __init__(self, in_size, vocabulary_size_sender, hidden_size, n_actions, lr=0.5):
        
        self.in_size = in_size
        self.vocabulary_size_sender = vocabulary_size_sender
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.lr = lr
        
        self.model = Receiver_Net(self.in_size, self.vocabulary_size_sender, self.hidden_size, self.n_actions, self.lr)
        
        self.training = True
        self.baseline = 0
        self.n = 0
        self.ent_reg = 0

    def loss(self, error, logprobs, entropy):

        # added regularization to decrease variance 

        #policy_loss = ((error.detach() - self.baseline)*(-logprobs)).mean()
        #entropy_loss = -entropy.mean() * self.ent_reg

        loss = (-logprobs * error.detach()).mean() #policy_loss + entropy_loss 

        if self.training:
            self.n += 1.
            #self.baseline += (error.detach().mean().item() - self.baseline) / self.n
        
        return loss