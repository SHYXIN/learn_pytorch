import torch.nn as nn
import torch.nn.functional as F


class CNN_network(nn.Module):
    def __init__(self):
        super(CNN_network, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16,5, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.linear1 = nn.Linear(32*32*16, 64)
        self.linear2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)

        x = x.view(-1, 32 * 32 * 16)
        x = F.relu(self.linear1(x))
        x = F.log_softmax(self.linear2(x), dim=1)

        return x
