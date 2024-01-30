# 1. Import the following libraries:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(0)
# 2. Read the previously prepared dataset, which should have been named
# dccc_prepared.csv.
data = pd.read_csv("dcc_prepared.csv")
print(data.head())

# 3. Separate the features from the target.
X = data.iloc[:, :-1]
y = data["default payment next month"]


# 4. Using scikit-learn's train_test_split function, split the dataset into training,
# validation, and testing sets. Use a 60:20:20 split ratio. Set random_state to 0.
X_new, X_test, y_new, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
dev_per = X_test.shape[0] / X_new.shape[0]
X_train, X_dev, y_train, y_dev = train_test_split(X_new, y_new, test_size=dev_per, random_state=0)

print("Training sets:", X_train.shape, y_train.shape)
print("Validation sets:", X_dev.shape, y_dev.shape)
print("Testing sets:", X_test.shape, y_test.shape)

# Training sets: (28036, 22) (28036,)
# Validation sets: (9346, 22) (9346,)
# Testing sets: (9346, 22) (9346,)

# 5. Convert the validation and testing sets into tensors, considering that the
# features' matrix should be of the float type, while the target matrix should
# not. Leave the training sets unconverted for the moment as they will undergo
# further transformations.
X_dev_torch = torch.tensor(X_dev.values).float()
y_dev_torch = torch.tensor(y_dev.values)
X_test_torch = torch.tensor(X_test.values).float()
y_test_torch = torch.tensor(y_test.values)

# 6. Build a custom module class for defining the layers of the network. Include a
# forward function that specifies the activation functions that will be applied to the
# output of each layer. Use ReLU for all layers except for the output, where you
# should use log_softmax.
class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden_1 = nn.Linear(input_size, 10)
        self.hidden_2 = nn.Linear(10, 10)
        self.hidden_3 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, x):
        z = F.relu(self.hidden_1(x))
        z = F.relu(self.hidden_2(z))
        z = F.relu(self.hidden_3(z))
        out = F.log_softmax(self.output(z), 1)

        return out

# 7. Instantiate the model and define all the variables required to train the model.
# Set the number of epochs to 50 and the batch size to 128. Use a learning rate
# of 0.001.
model = Classifier(X_train.shape[1])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
batch_size = 128

# 8. Train the network using the training set's data. Use the validation sets to
# measure performance. To do this, save the loss and the accuracy for both the
# training and validation sets in each epoch.
train_losses, dev_losses, train_acc, dev_acc = [], [], [], []

for e in range(epochs):
    X_, y_ = shuffle(X_train, y_train)
    running_loss = 0
    running_acc = 0
    iterations = 0

    for i in range(0, len(X_), batch_size):
        iterations += 1
        b = i + batch_size
        X_batch = torch.tensor(X_.iloc[i:b, :].values).float()
        y_batch = torch.tensor(y_.iloc[i:b].values)

        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        ps = torch.exp(pred)
        top_p, top_class = ps.topk(1, dim=1)
        running_acc += accuracy_score(y_batch, top_class)

    dev_loss = 0
    acc = 0

    # Turn off gradients for validation, saves memory and computation
    with torch.no_grad():
        pred_dev = model(X_dev_torch)
        dev_loss = criterion(pred_dev, y_dev_torch)

        ps_dev = torch.exp(pred_dev)
        top, top_class_dev = ps_dev.topk(1, dim=1)
        acc = accuracy_score(y_dev_torch, top_class_dev)

    train_losses.append(running_loss/iterations)
    dev_losses.append(dev_loss)
    train_acc.append(running_acc/iterations)
    dev_acc.append(acc)

    print("Epoch: {}/{}".format(e+1, epochs),
          "Training Loss: {:.3f}..".format(running_loss/iterations),
          "Validation Loss: {:.3f}..".format(dev_loss),
          "Trainning Accuracy: {:3f}..".format(running_acc/iterations),
          "Validation Accuracy: {:3f}..".format(acc))
# Epoch: 1/50 Training Loss: 0.671.. Validation Loss: 0.631.. Trainning Accuracy: 0.575852.. Validation Accuracy: 0.635138..
# Epoch: 2/50 Training Loss: 0.619.. Validation Loss: 0.612.. Trainning Accuracy: 0.662464.. Validation Accuracy: 0.662101..
# Epoch: 3/50 Training Loss: 0.608.. Validation Loss: 0.604.. Trainning Accuracy: 0.677592.. Validation Accuracy: 0.682217..
# Epoch: 4/50 Training Loss: 0.600.. Validation Loss: 0.598.. Trainning Accuracy: 0.690092.. Validation Accuracy: 0.689707..
# Epoch: 5/50 Training Loss: 0.594.. Validation Loss: 0.596.. Trainning Accuracy: 0.694815.. Validation Accuracy: 0.691954..
# Epoch: 6/50 Training Loss: 0.594.. Validation Loss: 0.595.. Trainning Accuracy: 0.692578.. Validation Accuracy: 0.689279..
# Epoch: 7/50 Training Loss: 0.590.. Validation Loss: 0.593.. Trainning Accuracy: 0.695810.. Validation Accuracy: 0.689600..
# Epoch: 8/50 Training Loss: 0.587.. Validation Loss: 0.589.. Trainning Accuracy: 0.696839.. Validation Accuracy: 0.690670..
# Epoch: 9/50 Training Loss: 0.584.. Validation Loss: 0.586.. Trainning Accuracy: 0.695703.. Validation Accuracy: 0.694736..
# Epoch: 10/50 Training Loss: 0.580.. Validation Loss: 0.585.. Trainning Accuracy: 0.702237.. Validation Accuracy: 0.699123..
# Epoch: 11/50 Training Loss: 0.578.. Validation Loss: 0.583.. Trainning Accuracy: 0.704439.. Validation Accuracy: 0.698160..
# Epoch: 12/50 Training Loss: 0.577.. Validation Loss: 0.582.. Trainning Accuracy: 0.705753.. Validation Accuracy: 0.695699..
# Epoch: 13/50 Training Loss: 0.574.. Validation Loss: 0.580.. Trainning Accuracy: 0.706250.. Validation Accuracy: 0.699444..
# Epoch: 14/50 Training Loss: 0.573.. Validation Loss: 0.578.. Trainning Accuracy: 0.706747.. Validation Accuracy: 0.701370..
# Epoch: 15/50 Training Loss: 0.572.. Validation Loss: 0.577.. Trainning Accuracy: 0.707599.. Validation Accuracy: 0.700728..
# Epoch: 16/50 Training Loss: 0.571.. Validation Loss: 0.577.. Trainning Accuracy: 0.710156.. Validation Accuracy: 0.700942..
# Epoch: 17/50 Training Loss: 0.571.. Validation Loss: 0.574.. Trainning Accuracy: 0.707741.. Validation Accuracy: 0.702547..
# Epoch: 18/50 Training Loss: 0.568.. Validation Loss: 0.573.. Trainning Accuracy: 0.710440.. Validation Accuracy: 0.703617..
# Epoch: 19/50 Training Loss: 0.569.. Validation Loss: 0.574.. Trainning Accuracy: 0.707386.. Validation Accuracy: 0.704579..
# Epoch: 20/50 Training Loss: 0.567.. Validation Loss: 0.574.. Trainning Accuracy: 0.711151.. Validation Accuracy: 0.705007..
# Epoch: 21/50 Training Loss: 0.568.. Validation Loss: 0.571.. Trainning Accuracy: 0.709446.. Validation Accuracy: 0.702226..
# Epoch: 22/50 Training Loss: 0.566.. Validation Loss: 0.573.. Trainning Accuracy: 0.712216.. Validation Accuracy: 0.702012..
# Epoch: 23/50 Training Loss: 0.567.. Validation Loss: 0.570.. Trainning Accuracy: 0.709943.. Validation Accuracy: 0.702654..
# Epoch: 24/50 Training Loss: 0.566.. Validation Loss: 0.570.. Trainning Accuracy: 0.709872.. Validation Accuracy: 0.704473..
# Epoch: 25/50 Training Loss: 0.564.. Validation Loss: 0.569.. Trainning Accuracy: 0.711541.. Validation Accuracy: 0.705756..
# Epoch: 26/50 Training Loss: 0.567.. Validation Loss: 0.569.. Trainning Accuracy: 0.709055.. Validation Accuracy: 0.706077..
# Epoch: 27/50 Training Loss: 0.563.. Validation Loss: 0.570.. Trainning Accuracy: 0.710973.. Validation Accuracy: 0.707682..
# Epoch: 28/50 Training Loss: 0.562.. Validation Loss: 0.569.. Trainning Accuracy: 0.713601.. Validation Accuracy: 0.704259..
# Epoch: 29/50 Training Loss: 0.564.. Validation Loss: 0.567.. Trainning Accuracy: 0.710724.. Validation Accuracy: 0.706933..
# Epoch: 30/50 Training Loss: 0.560.. Validation Loss: 0.568.. Trainning Accuracy: 0.714240.. Validation Accuracy: 0.706612..
# Epoch: 31/50 Training Loss: 0.563.. Validation Loss: 0.569.. Trainning Accuracy: 0.712749.. Validation Accuracy: 0.709501..
# Epoch: 32/50 Training Loss: 0.562.. Validation Loss: 0.565.. Trainning Accuracy: 0.710938.. Validation Accuracy: 0.706398..
# Epoch: 33/50 Training Loss: 0.561.. Validation Loss: 0.567.. Trainning Accuracy: 0.711364.. Validation Accuracy: 0.710464..
# Epoch: 34/50 Training Loss: 0.559.. Validation Loss: 0.568.. Trainning Accuracy: 0.712642.. Validation Accuracy: 0.709822..
# Epoch: 35/50 Training Loss: 0.560.. Validation Loss: 0.565.. Trainning Accuracy: 0.711967.. Validation Accuracy: 0.708645..
# Epoch: 36/50 Training Loss: 0.557.. Validation Loss: 0.565.. Trainning Accuracy: 0.714702.. Validation Accuracy: 0.707682..
# Epoch: 37/50 Training Loss: 0.559.. Validation Loss: 0.566.. Trainning Accuracy: 0.713139.. Validation Accuracy: 0.711641..
# Epoch: 38/50 Training Loss: 0.557.. Validation Loss: 0.566.. Trainning Accuracy: 0.714773.. Validation Accuracy: 0.711106..
# Epoch: 39/50 Training Loss: 0.558.. Validation Loss: 0.563.. Trainning Accuracy: 0.713778.. Validation Accuracy: 0.705756..
# Epoch: 40/50 Training Loss: 0.557.. Validation Loss: 0.566.. Trainning Accuracy: 0.713033.. Validation Accuracy: 0.711427..
# Epoch: 41/50 Training Loss: 0.557.. Validation Loss: 0.563.. Trainning Accuracy: 0.712216.. Validation Accuracy: 0.711320..
# Epoch: 42/50 Training Loss: 0.557.. Validation Loss: 0.564.. Trainning Accuracy: 0.711612.. Validation Accuracy: 0.705435..
# Epoch: 43/50 Training Loss: 0.557.. Validation Loss: 0.564.. Trainning Accuracy: 0.712962.. Validation Accuracy: 0.710357..
# Epoch: 44/50 Training Loss: 0.556.. Validation Loss: 0.563.. Trainning Accuracy: 0.714915.. Validation Accuracy: 0.708431..
# Epoch: 45/50 Training Loss: 0.556.. Validation Loss: 0.563.. Trainning Accuracy: 0.713459.. Validation Accuracy: 0.708645..
# Epoch: 46/50 Training Loss: 0.555.. Validation Loss: 0.562.. Trainning Accuracy: 0.715234.. Validation Accuracy: 0.706398..
# Epoch: 47/50 Training Loss: 0.555.. Validation Loss: 0.562.. Trainning Accuracy: 0.715945.. Validation Accuracy: 0.707789..
# Epoch: 48/50 Training Loss: 0.554.. Validation Loss: 0.571.. Trainning Accuracy: 0.715341.. Validation Accuracy: 0.711427..
# Epoch: 49/50 Training Loss: 0.554.. Validation Loss: 0.562.. Trainning Accuracy: 0.717969.. Validation Accuracy: 0.707682..
# Epoch: 50/50 Training Loss: 0.556.. Validation Loss: 0.565.. Trainning Accuracy: 0.713494.. Validation Accuracy: 0.710143..


# 9. Plot the loss of both sets.
fig = plt.figure(figsize=(15, 5))
plt.plot(train_losses, label="Training loss")
plt.plot(dev_losses, label="Validation loss")
plt.legend(frameon=False, fontsize=15)
plt.show()

# 10. Plot the accuracy of both sets.
fig = plt.figure(figsize=(15, 5))
plt.plot(train_acc, label="Training accuracy")
plt.plot(dev_acc, label="Validation accuracy")
plt.legend(frameon=False, fontsize=15)
plt.show()