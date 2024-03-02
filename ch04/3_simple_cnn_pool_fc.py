import torch.nn as nn
import torch.nn.functional as F


class CNN_network(nn.Module):

    def __init__(self):
        super(CNN_network, self).__init__()

        self.conv1 = nn.Conv2d(3, 18, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)


        self.linear1 = nn.Linear(32 * 32 * 16, 64)
        self.linear2 = nn.Linear(64, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = x.view(-1, 32 * 32 * 16)
        x = F.relu(self.linear1(x))
        x = F.log_softmax(self.linear2(x))

        return x

