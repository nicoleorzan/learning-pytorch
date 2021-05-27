import torch
import torch.nn.functional as F

def reward(actions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    rew = torch.zeros(actions.shape)

    for i in range(len(actions)):
        if (actions[i] == targets[i]):
            rew[i] = 1

    #rew = F.cross_entropy(actions, targets)

    return rew

def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    acc = torch.tensor(torch.sum(outputs == labels).item()/len(labels))
    return acc