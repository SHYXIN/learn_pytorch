# Activity 4.02: Implementing Data Augmentation
# 1. Duplicate the notebook from the previous activity


import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 2. Change the definition of the transform variable so that it includes,
# in addition to normalizing and converting the data into tensors, the
# following transformations:
# For the training/validation sets, a RandomHorizontalFlip function with a
# probability of 50% (0.5) and a RandomGrayscale function with a probability of
# 10% (0.1).
# For the testing set, do not add any other transformations
transform = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomGrayscale(0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# 3. Set a batch size of 100 images and download both the training and testing data
# from the CIFAR10 dataset:
batch_size = 100
train_data = datasets.CIFAR10('data2', train=True,
                              download=True,
                              transform=transform['train']
                              )
test_data = datasets.CIFAR10('data2', train=False,
                             download=True,
                             transform=transform['test'])

# The preceding code downloads both the training and testing datasets that
# are available through PyTorch's torchvision package. The datasets are
# transformed as per the transformations defined in the previous step

# 4. Using a validation size of 20%, define the training and validation sampler that will
# be used to divide the dataset into those two sets:
dev_size = 0.2
idx = list(range(len(train_data)))
np.random.shuffle(idx)
split_size = int(np.floor(dev_size * len(train_data)))
train_idx, dev_idx = idx[split_size:], idx[:split_size]

train_sampler = SubsetRandomSampler(train_idx)
dev_sampler = SubsetRandomSampler(dev_idx)

# In order to split the training set into two sets (training and validation), a list of
# indexes is defined for each of the sets, which can then be randomly sampled
# using the SubsetRandomSampler function.

# 5. Use the DataLoader() function to define the batches of each set of data to
# be used:
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           sampler=train_sampler)

dev_loader = torch.utils.data.DataLoader(train_data,
                                         batch_size,
                                         sampler=dev_sampler)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size)


# PyTorch's DataLoader function is used to allow the creation of batches that
# will be fed to the model during the training, validation, and testing phases of the
# development process.

# 6. Define the architecture of your network. Use the following information to do so:

# Conv1: A convolutional layer that takes the colored image as input and passes it
# through 10 filters of size 3. Both the padding and the stride should be set to 1.

# Conv2: A convolutional layer that passes the input data through 20 filters of size
# 3. Both the padding and the stride should be set to 1.

# Conv3: A convolutional layer that passes the input data through 40 filters of size
# 3. Both the padding and the stride should be set to 1.

# Use the ReLU activation function after each convolutional layer.

# Use a pooling layer after each convolutional layer, with a filter size and stride
# of 2.

# Use a dropout term set to 20%, after flattening the image.

# Linear1: A fully connected layer that receives the flattened matrix from the
# previous layer as input and generates an output of 100 units. Use the ReLU
# activation function for this layer. The dropout term here is set to 20%.

# Linear2: A fully connected layer that generates 10 outputs, one for each class
# label. Use the log_softmax activation function for the output layer:

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, 1, 1)
        self.conv2 = nn.Conv2d(10, 20, 3, 1, 1)
        self.conv3 = nn.Conv2d(20, 40, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)

        self.linear1 = nn.Linear(40 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 40 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.linear2(x), dim=1)

        return x


# The preceding code snippet consists of a class where the network architecture is
# defined (the __init__ method), as well as the steps that are followed during
# the forward pass of the information (the forward method).

# 7. Define all of the parameters that are required to train your model. Set the
# number of epochs to 50:
model = CNN()
# model = CNN().to("cuda")
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100
# The optimizer that we selected for this exercise is Adam. Also, the negative
# log-likelihood is used as the loss function, as in the previous chapter of this book.
# If your machine has a GPU available, the instantiation of the model should be
# done as follows:
# model = CNN().to("cuda")

# 8. Train your network and be sure to save the values for the loss and accuracy of
# both the training and validation sets:
train_losses, dev_losses, train_acc, dev_acc = [], [], [], []
x_axis = []

# For loop through the epochs
for e in range(1, epochs + 1):
    losses = 0
    acc = 0
    iterations = 0

    model.train()

    """
    For loop through the batches (created using the train loader)
    """
    for data, target in train_loader:
        iterations += 1

        # Forward and backward pass of the training data
        pred = model(data)
        loss = loss_function(pred, target)
        # 使用gpu
        # pred = model(data.to("cuda"))
        # loss = loss_function(pred, target.to("cuda"))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()
        p = torch.exp(pred)
        top_p, top_class = p.topk(1, dim=1)
        acc += accuracy_score(target, top_class)

    dev_losss = 0
    dev_accs = 0
    iter_2 = 0

    # Validation of model for given epoch
    if e % 2 == 0 or e == 1:
        x_axis.append(e)

        with torch.no_grad():
            model.eval()

            """
            For loop through the batches of the validation set
            """
            for data_dev, target_dev in dev_loader:
                iter_2 += 1

                dev_pred = model(data_dev)
                dev_loss = loss_function(dev_pred, target_dev)
                # 使用gpu
                # dev_pred = model(data_dev.to("cuda"))
                # dev_loss = loss_function(dev_pred, target_dev.to("cuda"))
                dev_losss += dev_loss.item()

                dev_p = torch.exp(dev_pred)
                top_p, dev_top_class = dev_p.topk(1, dim=1)
                dev_accs += accuracy_score(target_dev, dev_top_class)
                # 使用gpu
                # accuracy_score += accuracy_score(target_dev.to("cpu"),
                #                                  dev_top_class.to("cpu"))

        # Losses and accuracy are appended to be printed
        train_losses.append(losses / iterations)
        dev_losses.append(dev_losss / iter_2)
        train_acc.append(acc / iterations)
        dev_acc.append(dev_accs / iter_2)

        print("Epoch: {}/{}..".format(e, epochs),
              "Training Loss: {:.3f}..".format(losses / iterations),
              "Validation Loss: {:.3f}..".format(dev_losss / iter_2),
              "Training Accuracy: {:.3f}..".format(acc / iterations),
              "Validation Accuracy: {:.3f}..".format(dev_accs / iter_2))

# 9. Plot the loss and accuracy of both sets. To plot the loss, use the following code:
plt.plot(x_axis, train_losses, label='Training loss')
plt.plot(x_axis, dev_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

# To plot the accuracy, use the following code:
plt.plot(x_axis, train_acc, label='Training accuracy')
plt.plot(x_axis, dev_acc, label='Validation accuracy')
plt.legend(frameon=False)
plt.show()
# The plot should look similar to the following:
# As can be seen, after the 15th epoch, overfitting starts to affect the model.
# 10. Check the model's accuracy on the testing set:

model.eval()
iter_3 = 0
acc_test = 0

for data_test, target_test in test_loader:
    iter_3 += 1
    test_pred = model(data_test)
    test_pred = torch.exp(test_pred)
    top_p, top_class_test = test_pred.topk(1, dim=1)
    acc_test += accuracy_score(target_test, top_class_test)

print(acc_test / iter_3)
# The accuracy of the testing set is very similar to the accuracy that was achieved
# by the other two sets, which means that the model has the capability to perform
# equally well on unseen data; it should be around 72%.