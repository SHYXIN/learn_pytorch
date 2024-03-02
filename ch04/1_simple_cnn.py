import torch.nn as nn
import torch.nn.functional as F


class CNN_network(nn.Module):
    def __init__(self):
        super(CNN_network, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x


# When defining the convolutional layer, the arguments that are passed through from
# left to right refer to the input channels, output channels (number of filters), kernel
# size (filter size), stride, and padding

# Another valid approach, equivalent to the previous example, consists of a
# combination of the syntax from custom modules and the use of Sequential
# containers, as can be seen in the following code snippet:

import torch.nn as nn


class CNN_network2(nn.Module):
    def __init__(self):
        super(CNN_network2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 18, 3, 1, 1),
                                   nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        return x

# Here, the definition of layers occurs inside the Sequential container. Typically, one
# container includes a convolutional layer, an activation function, and a pooling layer. A
# new set of layers is included in a different container below it.
