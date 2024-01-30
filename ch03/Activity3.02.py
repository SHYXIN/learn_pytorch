# 1. Import the same libraries as in the previous activity.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)

# 2. Load the data and split the features from the target. Next, split the data into
# three subsets (training, validation, and testing) using a 60:20:20 split ratio. Finally,
# convert the validation and testing sets into PyTorch tensors, just as you did in the
# previous activity.
data = pd.read_csv("dcc_prepared.csv")
print(data.head())

X = data.iloc[:, :-1]
y = data['default payment next month']

X_new, X_test, y_new, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
dev_per = X_test.shape[0] / X_new.shape[0]
X_train, X_dev, y_train, y_dev = train_test_split(X_new, y_new, test_size=dev_per, random_state=0)

X_dev_torch = torch.tensor(X_dev.values).float()
y_dev_torch = torch.tensor(y_dev.values)
X_test_torch = torch.tensor(X_test.values).float()
y_test_torch = torch.tensor(y_test.values)


# First fine-tuning approach
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
        out = F.log_softmax(self.output(z), dim=1)

        return out


model = Classifier(X_train.shape[1])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000  # 加大训练的批次
batch_size = 128

train_losses, dev_losses, train_acc, dev_acc = [], [], [], []
x_axis = []

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

        log_ps = model(X_batch)
        loss = criterion(log_ps, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        running_acc += accuracy_score(y_batch, top_class)

    dev_loss = 0
    acc = 0

    # Turn off gradient for validations, saves memory and computation
    with torch.no_grad():
        log_dev = model(X_dev_torch)
        dev_loss = criterion(log_dev, y_dev_torch)

        ps_dev = torch.exp(log_dev)
        top_p, top_class_dev = ps_dev.topk(1, dim=1)
        acc = accuracy_score(y_dev_torch, top_class_dev)

    if e % 50 == 0 or e == 1:
        x_axis.append(e)

        train_losses.append(running_loss / iterations)
        dev_losses.append(dev_loss)
        train_acc.append(running_acc / iterations)
        dev_acc.append(acc)

        print("Epoch: {}/{}..".format(e, epochs),
              "Training Loss: {:.3f}..".format(running_loss / iterations),
              "Validation Loss: {:.3f}..".format(dev_loss),
              "Training Accuracy: {:.3f}..".format(running_loss / iterations),
              "Validation Accuracy: {:.3f}..".format(acc))

# Epoch: 1/1000..  Training Loss: 0.671..  Validation Loss: 0.628..  Training Accuracy: 0.576..  Validation Accuracy: 0.639
# Epoch: 50/1000..  Training Loss: 0.558..  Validation Loss: 0.567..  Training Accuracy: 0.713..  Validation Accuracy: 0.710
# Epoch: 100/1000..  Training Loss: 0.552..  Validation Loss: 0.563..  Training Accuracy: 0.717..  Validation Accuracy: 0.704
# Epoch: 150/1000..  Training Loss: 0.550..  Validation Loss: 0.561..  Training Accuracy: 0.719..  Validation Accuracy: 0.706
# Epoch: 200/1000..  Training Loss: 0.546..  Validation Loss: 0.562..  Training Accuracy: 0.722..  Validation Accuracy: 0.706
# Epoch: 250/1000..  Training Loss: 0.545..  Validation Loss: 0.558..  Training Accuracy: 0.722..  Validation Accuracy: 0.711
# Epoch: 300/1000..  Training Loss: 0.545..  Validation Loss: 0.557..  Training Accuracy: 0.721..  Validation Accuracy: 0.714
# Epoch: 350/1000..  Training Loss: 0.542..  Validation Loss: 0.558..  Training Accuracy: 0.726..  Validation Accuracy: 0.710
# Epoch: 400/1000..  Training Loss: 0.542..  Validation Loss: 0.558..  Training Accuracy: 0.723..  Validation Accuracy: 0.713
# Epoch: 450/1000..  Training Loss: 0.540..  Validation Loss: 0.557..  Training Accuracy: 0.727..  Validation Accuracy: 0.714
# Epoch: 500/1000..  Training Loss: 0.540..  Validation Loss: 0.558..  Training Accuracy: 0.727..  Validation Accuracy: 0.707
# Epoch: 550/1000..  Training Loss: 0.541..  Validation Loss: 0.565..  Training Accuracy: 0.724..  Validation Accuracy: 0.704
# Epoch: 600/1000..  Training Loss: 0.542..  Validation Loss: 0.554..  Training Accuracy: 0.723..  Validation Accuracy: 0.712
# Epoch: 650/1000..  Training Loss: 0.540..  Validation Loss: 0.558..  Training Accuracy: 0.726..  Validation Accuracy: 0.710
# Epoch: 700/1000..  Training Loss: 0.538..  Validation Loss: 0.554..  Training Accuracy: 0.725..  Validation Accuracy: 0.712
# Epoch: 750/1000..  Training Loss: 0.540..  Validation Loss: 0.557..  Training Accuracy: 0.725..  Validation Accuracy: 0.711
# Epoch: 800/1000..  Training Loss: 0.537..  Validation Loss: 0.554..  Training Accuracy: 0.729..  Validation Accuracy: 0.710
# Epoch: 850/1000..  Training Loss: 0.537..  Validation Loss: 0.557..  Training Accuracy: 0.728..  Validation Accuracy: 0.712
# Epoch: 900/1000..  Training Loss: 0.539..  Validation Loss: 0.554..  Training Accuracy: 0.726..  Validation Accuracy: 0.715
# Epoch: 950/1000..  Training Loss: 0.538..  Validation Loss: 0.555..  Training Accuracy: 0.727..  Validation Accuracy: 0.709
# Epoch: 1000/1000..  Training Loss: 0.536..  Validation Loss: 0.553..  Training Accuracy: 0.728..  Validation Accuracy: 0.711

# 3. Considering that the model is suffering from a high bias, the focus should be on
# increasing the number of epochs or increasing the size of the network by adding
# additional layers or units to each layer. The aim should be to approximate the
# accuracy over the testing set to 80%.
fig = plt.figure(figsize=(15, 5))
plt.plot(x_axis, train_losses, label='Training loss')
plt.plot(x_axis, dev_losses, label='Validation loss')
plt.legend(frameon=False, fontsize=15)
plt.show()

fig = plt.figure(figsize=(15, 5))
plt.plot(x_axis, train_acc, label="Training accuracy")
plt.plot(x_axis, dev_acc, label='Validation accuracy')
plt.legend(frameon=False, fontsize=15)
plt.show()


# Second fine-tuning approach
class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden_1 = nn.Linear(input_size, 10)
        self.hidden_2 = nn.Linear(10, 10)
        self.hidden_3 = nn.Linear(10, 10)
        self.hidden_4 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, x):
        z = F.relu(self.hidden_1(x))
        z = F.relu(self.hidden_2(z))
        z = F.relu(self.hidden_2(z))
        z = F.relu(self.hidden_4(z))
        out = F.log_softmax(self.output(z), dim=1)

        return out

model = Classifier(X_train.shape[1])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
batch_size = 128


train_losses, dev_losses, train_acc, dev_acc = [], [], [], []
x_axis = []

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

        log_ps = model(X_batch)
        loss = criterion(log_ps, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        running_acc += accuracy_score(y_batch, top_class)

    dev_loss = 0
    acc = 0

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        log_dev = model(X_dev_torch)
        dev_loss = criterion(log_dev, y_dev_torch)

        ps_dev = torch.exp(log_dev)
        top_p, top_class_dev = ps_dev.topk(1, dim=1)
        acc = accuracy_score(y_batch, top_class_dev)

    if e % 50 == 0 or e == 1:
        x_axis.append(e)

        train_losses.append(running_loss/iterations)
        dev_losses.append(dev_loss)
        train_acc.append(running_acc/iterations)
        dev_acc.append(acc)

        print("Epoch: {}/{}".format(e, epochs),
              "Training Loss: {:.3f}..".format(running_loss/iterations),
              "Validation Loss: {:3f}..".format(dev_loss),
              "Training Accuracy: {:3f}..".format(running_acc/iterations),
              "Validation Accuracy: {:3f}..".format(acc))

fig = plt.figure(figsize=(15, 5))
plt.plot(x_axis, train_losses, label="Training loss")
plt.plot(x_axis, dev_losses, label="Validation loss")
plt.legend(frameon=False, fontsize=15)
plt.show()


fig = plt.figure(figsize=(15, 5))
plt.plot(x_axis, train_acc, label='Training accuracy')
plt.plot(x_axis, dev_acc, label='Validation accuracy')
plt.show()

# Third fine-tuning approach
class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden_1 = nn.Linear(input_size, 50)
        self.hidden_2 = nn.Linear(50, 50)
        self.hidden_3 = nn.Linear(50, 50)
        self.hidden_4 = nn.Linear(50, 50)
        self.output = nn.Linear(50, 2)

    def forward(self, x):
        z = F.relu(self.hidden_1(x))
        z = F.relu(self.hidden_2(z))
        z = F.relu(self.hidden_3(z))
        z = F.relu(self.hidden_4(z))
        out = F.log_softmax(self.output(z), dim=1)
        return out

model = Classifier(X_train.shape[1])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
batch_size = 128

train_losses, dev_losses, train_acc, dev_acc = [], [], [], []
x_axis = []

for e in range(epochs):
    X_, y_ = shuffle(X_train, y_train)
    running_loss = 0
    running_acc = 0
    iterations = 0

    for i in range(0, len(X_), batch_size):
        iterations += 1
        b = i + iterations
        X_batch = torch.tensor(X_.iloc[i:b, :].vlaues).float()
        y_batch = torch.tensor(y_.iloc[i:b].values)

        log_ps = model(X_batch)
        loss = criterion(log_ps, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        running_acc += accuracy_score(y_batch, top_class)

    dev_loss = 0
    acc = 0

    # Turn off gradient for validation, saves memory and computation
    with torch.no_grad():
        log_dev = model(X_dev_torch)
        dev_loss = criterion(log_dev, y_dev_torch)

        ps_dev = torch.exp(log_dev)
        top_p, top_class_dev = ps_dev.topk(1, dim=1)
        acc = accuracy_score(y_dev_torch, top_class_dev)

    if e % 50 == 0 or e == 1:
        x_axis.append(e)

        train_losses.append(running_loss/iterations)
        dev_losses.append(dev_loss)
        train_acc.append(running_acc/iterations)
        dev_acc.append(acc)

        print("Epochs: {}/{}..".format(e, epochs),
              "Training Loss: {:.3f}..".format(running_loss/iterations),
              "Validation Loss: {:.3f}..".format(dev_loss),
              "Training Accuracy: {:.3f}..".format(running_acc/iterations),
              "Validation Accuracy: {:.3f}..".format(acc))

# Epoch: 1/1000..  Training Loss: 0.626..  Validation Loss: 0.595..  Training Accuracy: 0.644..  Validation Accuracy: 0.689
# Epoch: 50/1000..  Training Loss: 0.530..  Validation Loss: 0.557..  Training Accuracy: 0.732..  Validation Accuracy: 0.715
# Epoch: 100/1000..  Training Loss: 0.497..  Validation Loss: 0.553..  Training Accuracy: 0.754..  Validation Accuracy: 0.718
# Epoch: 150/1000..  Training Loss: 0.458..  Validation Loss: 0.576..  Training Accuracy: 0.778..  Validation Accuracy: 0.723
# Epoch: 200/1000..  Training Loss: 0.434..  Validation Loss: 0.583..  Training Accuracy: 0.793..  Validation Accuracy: 0.737
# Epoch: 250/1000..  Training Loss: 0.440..  Validation Loss: 0.591..  Training Accuracy: 0.790..  Validation Accuracy: 0.741
# Epoch: 300/1000..  Training Loss: 0.381..  Validation Loss: 0.624..  Training Accuracy: 0.821..  Validation Accuracy: 0.757
# Epoch: 350/1000..  Training Loss: 0.369..  Validation Loss: 0.653..  Training Accuracy: 0.826..  Validation Accuracy: 0.754
# Epoch: 400/1000..  Training Loss: 0.347..  Validation Loss: 0.671..  Training Accuracy: 0.839..  Validation Accuracy: 0.756
# Epoch: 450/1000..  Training Loss: 0.334..  Validation Loss: 0.723..  Training Accuracy: 0.846..  Validation Accuracy: 0.765
# Epoch: 500/1000..  Training Loss: 0.326..  Validation Loss: 0.750..  Training Accuracy: 0.850..  Validation Accuracy: 0.771
# Epoch: 550/1000..  Training Loss: 0.306..  Validation Loss: 0.754..  Training Accuracy: 0.858..  Validation Accuracy: 0.774
# Epoch: 600/1000..  Training Loss: 0.348..  Validation Loss: 0.795..  Training Accuracy: 0.845..  Validation Accuracy: 0.766
# Epoch: 650/1000..  Training Loss: 0.287..  Validation Loss: 0.825..  Training Accuracy: 0.868..  Validation Accuracy: 0.778
# Epoch: 700/1000..  Training Loss: 0.280..  Validation Loss: 0.933..  Training Accuracy: 0.870..  Validation Accuracy: 0.770
# Epoch: 750/1000..  Training Loss: 0.273..  Validation Loss: 0.876..  Training Accuracy: 0.874..  Validation Accuracy: 0.782
# Epoch: 800/1000..  Training Loss: 0.278..  Validation Loss: 0.984..  Training Accuracy: 0.875..  Validation Accuracy: 0.761
# Epoch: 850/1000..  Training Loss: 0.274..  Validation Loss: 0.896..  Training Accuracy: 0.876..  Validation Accuracy: 0.790
# Epoch: 900/1000..  Training Loss: 0.247..  Validation Loss: 0.919..  Training Accuracy: 0.887..  Validation Accuracy: 0.789
# Epoch: 950/1000..  Training Loss: 0.243..  Validation Loss: 0.960..  Training Accuracy: 0.890..  Validation Accuracy: 0.793
# Epoch: 1000/1000..  Training Loss: 0.244..  Validation Loss: 1.036..  Training Accuracy: 0.891..  Validation Accuracy: 0.787

fig = plt.figure(figsize=(15, 5))
plt.plot(x_axis, train_losses, label='Train loss')
plt.plot(x_axis, dev_losses, label='Validation loss')
plt.legend(frameon=False, fontsize=15)
plt.show()

fig = plt.figure(figsize=(15, 5))
plt.plot(x_axis, train_acc, label='Training accuracy')
plt.plot(x_axis, dev_acc, label='Validation accuracy')
plt.legend(frameon=False, fontsize=15)
plt.show()


# Fourth fine-tuning approach
class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden_1 = nn.Linear(input_size, 50)
        self.hidden_2 = nn.Linear(50, 50)
        self.hidden_3 = nn.Linear(50, 50)
        self.hidden_4 = nn.Linear(50, 50)
        self.output = nn.Linear(50, 2)

    def forward(self, x):
        z = F.relu(self.hidden_1(x))
        z = F.relu(self.hidden_2(z))
        z = F.relu(self.hidden_3(z))
        z = F.relu(self.hidden_4(z))
        out = F.log_softmax(self.output(z))

        return out

model = Classifier(X_train.shape[1])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 2000
batch_size = 128

for e in range(1, epochs + 1):
    X_, y_ = shuffle(X_train, y_train)
    running_loss = 0
    running_acc = 0
    iterations = 0

    for i in range(len(X_), batch_size):
        iterations += 1
        b = i + batch_size
        X_batch = torch.tensor(X_.iloc[i:b, :].values).float()
        y_batch = torch.tensor(y_.iloc[i:b].values)

        log_ps = model(X_batch)
        loss = criterion(log_ps, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        running_acc += accuracy_score(y_batch, top_class)

    dev_loss = 0
    acc = 0

    model.eval()  # 进入评估模式

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        log_dev = model(X_dev_torch)
        dev_loss = criterion(log_dev, y_dev_torch)

        ps_dev = torch.exp(log_dev)
        top_p, top_clas_dev = ps_dev.topk(1, dim=1)
        acc = accuracy_score(y_dev_torch, top_class_dev)

    model.train()

    if e % 50 == 0 or e == 1:
        x_axis.append(e)

        train_losses.append(running_loss/iterations)
        dev_losses.append(dev_loss)
        train_acc.append(running_acc/iterations)
        dev_acc.append(acc)

        print("Epochs: {}/{}..".format(e, epochs),
              "Training Loss: {:.3f}..".format(running_loss/iterations),
              "Validation Loss: {:.3f}..".format(dev_loss),
              "Training Accuracy: {:.3f}..".format(running_acc/iterations),
              "Validation Accuracy: {:3f}..".format(acc))


# Epoch: 1/2000..  Training Loss: 0.625..  Validation Loss: 0.594..  Training Accuracy: 0.654..  Validation Accuracy: 0.690
# Epoch: 50/2000..  Training Loss: 0.543..  Validation Loss: 0.552..  Training Accuracy: 0.724..  Validation Accuracy: 0.714
# Epoch: 100/2000..  Training Loss: 0.529..  Validation Loss: 0.545..  Training Accuracy: 0.732..  Validation Accuracy: 0.718
# Epoch: 150/2000..  Training Loss: 0.516..  Validation Loss: 0.542..  Training Accuracy: 0.740..  Validation Accuracy: 0.720
# Epoch: 200/2000..  Training Loss: 0.511..  Validation Loss: 0.538..  Training Accuracy: 0.744..  Validation Accuracy: 0.725
# Epoch: 250/2000..  Training Loss: 0.504..  Validation Loss: 0.535..  Training Accuracy: 0.750..  Validation Accuracy: 0.726
# Epoch: 300/2000..  Training Loss: 0.501..  Validation Loss: 0.532..  Training Accuracy: 0.750..  Validation Accuracy: 0.729
# Epoch: 350/2000..  Training Loss: 0.497..  Validation Loss: 0.530..  Training Accuracy: 0.753..  Validation Accuracy: 0.727
# Epoch: 400/2000..  Training Loss: 0.495..  Validation Loss: 0.528..  Training Accuracy: 0.758..  Validation Accuracy: 0.734
# Epoch: 450/2000..  Training Loss: 0.491..  Validation Loss: 0.524..  Training Accuracy: 0.758..  Validation Accuracy: 0.736
# Epoch: 500/2000..  Training Loss: 0.485..  Validation Loss: 0.522..  Training Accuracy: 0.760..  Validation Accuracy: 0.741
# Epoch: 550/2000..  Training Loss: 0.488..  Validation Loss: 0.526..  Training Accuracy: 0.760..  Validation Accuracy: 0.734
# Epoch: 600/2000..  Training Loss: 0.484..  Validation Loss: 0.523..  Training Accuracy: 0.761..  Validation Accuracy: 0.737
# Epoch: 650/2000..  Training Loss: 0.483..  Validation Loss: 0.520..  Training Accuracy: 0.761..  Validation Accuracy: 0.741
# Epoch: 700/2000..  Training Loss: 0.479..  Validation Loss: 0.522..  Training Accuracy: 0.764..  Validation Accuracy: 0.738
# Epoch: 750/2000..  Training Loss: 0.477..  Validation Loss: 0.521..  Training Accuracy: 0.768..  Validation Accuracy: 0.742
# Epoch: 800/2000..  Training Loss: 0.482..  Validation Loss: 0.521..  Training Accuracy: 0.763..  Validation Accuracy: 0.741
# Epoch: 850/2000..  Training Loss: 0.478..  Validation Loss: 0.519..  Training Accuracy: 0.765..  Validation Accuracy: 0.744
# Epoch: 900/2000..  Training Loss: 0.473..  Validation Loss: 0.518..  Training Accuracy: 0.768..  Validation Accuracy: 0.748
# Epoch: 950/2000..  Training Loss: 0.467..  Validation Loss: 0.520..  Training Accuracy: 0.775..  Validation Accuracy: 0.748
# Epoch: 1000/2000..  Training Loss: 0.475..  Validation Loss: 0.517..  Training Accuracy: 0.767..  Validation Accuracy: 0.750
# Epoch: 1050/2000..  Training Loss: 0.471..  Validation Loss: 0.518..  Training Accuracy: 0.773..  Validation Accuracy: 0.745
# Epoch: 1100/2000..  Training Loss: 0.469..  Validation Loss: 0.518..  Training Accuracy: 0.775..  Validation Accuracy: 0.746
# Epoch: 1150/2000..  Training Loss: 0.470..  Validation Loss: 0.517..  Training Accuracy: 0.774..  Validation Accuracy: 0.748
# Epoch: 1200/2000..  Training Loss: 0.470..  Validation Loss: 0.514..  Training Accuracy: 0.772..  Validation Accuracy: 0.751
# Epoch: 1250/2000..  Training Loss: 0.469..  Validation Loss: 0.517..  Training Accuracy: 0.775..  Validation Accuracy: 0.749
# Epoch: 1300/2000..  Training Loss: 0.470..  Validation Loss: 0.515..  Training Accuracy: 0.770..  Validation Accuracy: 0.748
# Epoch: 1350/2000..  Training Loss: 0.466..  Validation Loss: 0.518..  Training Accuracy: 0.774..  Validation Accuracy: 0.748
# Epoch: 1400/2000..  Training Loss: 0.468..  Validation Loss: 0.514..  Training Accuracy: 0.773..  Validation Accuracy: 0.752
# Epoch: 1450/2000..  Training Loss: 0.464..  Validation Loss: 0.515..  Training Accuracy: 0.777..  Validation Accuracy: 0.751
# Epoch: 1500/2000..  Training Loss: 0.462..  Validation Loss: 0.510..  Training Accuracy: 0.779..  Validation Accuracy: 0.754
# Epoch: 1550/2000..  Training Loss: 0.462..  Validation Loss: 0.513..  Training Accuracy: 0.778..  Validation Accuracy: 0.751
# Epoch: 1600/2000..  Training Loss: 0.464..  Validation Loss: 0.508..  Training Accuracy: 0.776..  Validation Accuracy: 0.753
# Epoch: 1650/2000..  Training Loss: 0.464..  Validation Loss: 0.515..  Training Accuracy: 0.775..  Validation Accuracy: 0.752
# Epoch: 1700/2000..  Training Loss: 0.459..  Validation Loss: 0.513..  Training Accuracy: 0.778..  Validation Accuracy: 0.755
# Epoch: 1750/2000..  Training Loss: 0.464..  Validation Loss: 0.512..  Training Accuracy: 0.777..  Validation Accuracy: 0.755
# Epoch: 1800/2000..  Training Loss: 0.458..  Validation Loss: 0.513..  Training Accuracy: 0.778..  Validation Accuracy: 0.756
# Epoch: 1850/2000..  Training Loss: 0.463..  Validation Loss: 0.510..  Training Accuracy: 0.781..  Validation Accuracy: 0.756
# Epoch: 1900/2000..  Training Loss: 0.462..  Validation Loss: 0.513..  Training Accuracy: 0.779..  Validation Accuracy: 0.757
# Epoch: 1950/2000..  Training Loss: 0.457..  Validation Loss: 0.512..  Training Accuracy: 0.782..  Validation Accuracy: 0.755
# Epoch: 2000/2000..  Training Loss: 0.454..  Validation Loss: 0.511..  Training Accuracy: 0.781..  Validation Accuracy: 0.762

fig = plt.figure(figsize=(15, 5))
plt.plot(x_axis, train_losses, label='Training loss')
plt.plot(x_axis, dev_losses, label='Validation loss')
plt.legend(frameon=False, fontsize=15)
plt.show()


fig = plt.figure(figsize=(15, 5))
plt.plot(x_axis, train_acc, label='Training accuracy')
plt.plot(x_axis, dev_acc, label='Validation accuracy')
plt.legend(frameon=False, fontsize=15)
plt.show()

# After several fine-tuning approaches
class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden_1 = nn.Linear(input_size, 100)
        self.hidden_2 = nn.Linear(100, 100)
        self.hidden_3 = nn.Linear(100, 50)
        self.hidden_4 = nn.Linear(50, 50)
        self.output = nn.Linear(50, 2)

    def forward(self, x):
        z = F.relu(self.hidden_1(x))
        z = F.relu(self.hidden_2(z))
        z = F.relu(self.hidden_3(z))
        z = F.relu(self.hidden_4(z))
        out = F.log_softmax(self.output(z), dim=1)

        return out


model = Classifier(X_train.shape[1])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 4000
batch_size = 128

train_losses, dev_losses, train_acc, dev_acc = [], [], [], []
x_axis = []

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

        log_ps = model(X_batch)
        loss = criterion(log_ps, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        running_acc += accuracy_score(y_batch, top_class)

    dev_loss = 0
    acc = 0

    # Turn off gradients for validation, saves memory and computations
    model.eval()
    with torch.no_grad():
        log_dev = model(X_dev_torch)
        dev_loss = criterion(log_dev, y_dev_torch)

        ps_dev = torch.exp(log_dev)
        top_p, top_class_dev = ps_dev.topk(1, dim=1)
        acc = accuracy_score(y_dev_torch, top_class_dev)

    model.train()

    if i % 50 == 0 or e == 1:
        x_axis.append(e)

        train_losses.append(running_loss/iterations)
        dev_losses.append(dev_loss)
        train_acc.append(running_acc/iterations)
        dev_acc.append(acc)

        print("Epochs: {}/{}..".format(e, epochs),
              "Training Loss: {:.3f}..".format(running_loss/iterations),
              "Validation Loss: {:.3f}..".format(dev_loss),
              "Training accuracy: {:.3f}..".format(running_acc/iterations),
              "Validation accuracy: {:.3f}..".format(acc))


fig = plt.figure(figsize=(15, 5))
plt.plot(x_axis, train_losses, label='Training loss')
plt.plot(x_axis, dev_losses, label='Validation loss')
plt.legend(frameon=False, fontsize=15)
plt.show()


fig = plt.figure(figsize=(15, 5))
plt.plot(x_axis, train_acc, label='Training accuracy')
plt.plot(x_axis, dev_acc, label='Validation accuracy')
plt.legend(frameon=False, fontsize=15)
plt.show()

# 5. Using the best-performing model, perform a prediction over the testing set
# (which should not have been used during the fine-tuning process). Compare the
# prediction with the ground truth by calculating the accuracy of the model over
# this set:
model.eval()
test_pred = model(X_test_torch)
test_pred = torch.exp(test_pred)
top_p, top_class_test = test_pred.topk(1, dim=1)
acc_test = accuracy_score(y_test_torch, top_class_test)
print(accuracy_score)

# 0.8079392253370425


