# 1. Import the required libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim
import torch.nn.functional as F
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.manual_seed(0)

# 2. Load the dataset containing a set of 1,000 product reviews from Amazon,
# which is paired with a label of 0 (for negative reviews) or 1 (for positive reviews).
# Separate the data into two variables – one containing the reviews and the other
# containing the labels
data = pd.read_csv("data/amazon_cells_labelled.txt", sep="\t", header=None)
reviews = data.iloc[:, 0].str.lower()
sentiment = data.iloc[:, 1].values

# 3. Remove the punctuation from the reviews:
for i in punctuation:
    reviews = reviews.str.replace(i, "")

# 4. Create a variable containing the vocabulary of the entire set of reviews.
# Additionally, create a dictionary that maps each word to an integer, where the
# words will be the keys and the integers will be the values:
words = " ".join(reviews)
words = words.split()
vocabulary = set(words)
indexer = {word: index for (index, word) in enumerate(vocabulary)}

# 5. Encode the reviews data by replacing each word in a review with its
# paired integer:
indexed_reviews = []
for review in reviews:
    indexed_reviews.append([indexer[word] for word in review.split()])


# 6. Create a class containing the architecture of the network. Make sure that you
# include an embedding layer:
class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.output(out)
        out = out[-1, 0]
        out = torch.sigmoid(out).unsqueeze(0)

        return out


# The class contains an __init__ method, which defines the network
# architecture, and a forward method, which determines the way in which the
# data flows through the different layers.

# 7. Instantiate the model using 64 embedding dimensions and 128 neurons for
# three LSTM layers:
model = LSTM(len(vocabulary), 64, 128, 3)
print(model)

# LSTM(
#   (embedding): Embedding(1905, 64)
#   (lstm): LSTM(64, 128, num_layers=3, batch_first=True)
#   (output): Linear(in_features=128, out_features=1, bias=True)
# )

# 8. Define the loss function, an optimization algorithm, and the number of epochs
# to train for. For example, you can use the binary cross-entropy loss as the loss
# function, the Adam optimizer, and train for 10 epochs:
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

# 9. Create a for loop that goes through the different epochs and through every
# single review individually. For each review, perform a prediction, calculate the
# loss function, and update the parameters of the network. Additionally, calculate
# the accuracy of the network on that training data:
losses = []
acc = []
for e in range(1, epochs + 1):
    single_loss = []
    preds = []
    targets = []
    for i, r in enumerate(indexed_reviews):
        if len(r) <= 1:
            continue
        x = torch.Tensor([r]).long()
        y = torch.Tensor([sentiment[i]])

        pred = model(x)
        loss = loss_function(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_pred = np.round(pred.detach().numpy())
        preds.append(final_pred)
        targets.append(y)
        single_loss.append(loss.item())

    losses.append(np.mean(single_loss))
    binary_targets = [t.item() for t in targets]
    accuracy = accuracy_score(binary_targets, preds)
    # accuracy = accuracy_score(targets, preds)
    acc.append(accuracy)

    if e % 1 == 0:
        print("Epoch: ", e, "... Loss function: ", losses[-1], "... Accuracy: ", acc[-1])

# As in the previous activities, the training process consists of making a prediction,
# comparing it with the ground truth to calculate the loss function, and performing
# a backward pass to minimize the loss function

# 10. Plot the progress of the loss and accuracy over time. The following code is used
# to plot the loss function:
x_range = range(len(losses))
plt.plot(x_range, losses)
plt.xlabel("epochs")
plt.ylabel("Loss function")
plt.show()

# The following code is used to plot the accuracy score
x_range = range(len(acc))
plt.plot(x_range, acc)
plt.xlabel("epochs")
plt.ylabel("Accuracy score")
plt.show()
