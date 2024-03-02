# To load a dataset from PyTorch, use the following code. Besides downloading the
# dataset, the following code shows how to use data loaders to save resources by
# loading the images in batches, rather than all at once:
from torchvision import datasets
import torchvision.transforms as transforms


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])

# The transform variable is used to define the set of transformations to perform
# on the dataset. In this case, the dataset will be both converted into tensors and
# normalized in all its dimensions.
train_data = datasets.MNIST(root='data', train=True,
                            download=True, transform=transform)

test_data = datasets.MNIST(root='data', train=False,
                           download=True, transform=transform)

# In the preceding code, the dataset to be downloaded is MNIST. This is a popular
# dataset that contains images of hand-written grayscale numbers from zero to nine.
# PyTorch datasets provide both training and testing sets.

# As can be seen in the preceding snippet, to download the dataset, it is necessary
# to define the root of the data, which, by default, should be defined as data. Next,
# define whether you are downloading the training or the testing dataset. We set the
# download argument to True. Finally, we use the transform variable that we
# defined previously to perform the transformations on the datasets:
import numpy as np

dev_size = 0.2
idx = list(range(len(train_data)))
np.random.shuffle(idx)
split_size = int(np.floor(dev_size * len(train_data)))
train_idx, dev_idx = idx[split_size:], idx[:split_size]

# Considering that we need a third set of data (the validation set), the preceding
# code snippet is used to partition the training set into two sets. First, the size of the
# validation set is defined, and then the list of indexes that will be used for each of the
# datasets are defined (the training and the validation sets)
from torch.utils.data import SubsetRandomSampler
train_sampler = SubsetRandomSampler(train_idx)
dev_sampler = SubsetRandomSampler(dev_idx)

# In the preceding snippet, the SubsetRandomSampler() function from PyTorch is
# used to divide the original training set into training and validation sets by randomly
# sampling indexes. This will be used in the following step to generate the batches that
# will be fed into the model in each iteration:
import torch
batch_size = 20
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           sampler=train_sampler)
dev_loader = torch.utils.data.DataLoader(train_data,
                                         batch_size=batch_size,
                                         sampler=dev_sampler)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size)

# The DataLoader() function is used to load the images in batches, for each of the
# sets of data. First, the variable containing the set is passed as an argument and then
# the batch size is defined. Finally, the samplers that we created in the preceding step
# are used to make sure that the batches that are used in each iteration are randomly
# created, which helps improve the performance of the model. The resulting variables
# (train_loader, dev_loader, and test_loader) of this function will contain
# the values for the features and the target separately.

# The more complex the problem and the deeper the network, the longer it
# takes for the model to train. Considering this, the activities in this chapter
# may take longer than the ones in previous chapters.
