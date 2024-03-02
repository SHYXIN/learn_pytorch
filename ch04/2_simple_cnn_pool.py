# Using the same coding examples as before, the PyTorch way to define pooling layers
# is shown in the following code snippet:
import torch.nn as nn
import torch.nn.functional as F


class CNN_network(nn.Module):

    def __init__(self):
        super(CNN_network, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 3, 1,1)
        self.pool1 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        return x

# As can be seen, a pooling layer (MaxPool2d) was added to the network architecture
# in the __init__ method. Here, the arguments that go into the max pooling layers,
# from left to right, are the size of the filter (2) and the stride (2). Next, the forward
# method was updated to pass the information through the new pooling layer.
# Again, an equally valid approach is shown here, with the use of custom modules and
# Sequential containers:
import torch.nn as nn

class CNN_network2(nn.Module):
    def __init__(self):
        super(CNN_network2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 18, 3,1 ,1),
                                   nn.ReLU(),
                                   nn.MaxPool2d())

    def forward(self, x):
        x = self.conv1(x)
        return x

# As we mentioned previously, the pooling layer is also included in the same container
# as the convolutional layer, below the activation function. A subsequent set of
# layers (convolutional, activation, and pooling) would be defined below, in a new
# Sequential container.
# Again, the forward method no longer needs to call each layer individually; instead,
# it passes the information through the container, which holds both the layers and the
# activation function.
