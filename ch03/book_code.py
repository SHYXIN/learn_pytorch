# Take, for instance, the following code snippet of a two-layer neural network that's
# been defined using the Sequential container:
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(D_i, D_h),
                      nn.ReLU(),
                      nn.Linear(D_h, D_o),
                      nn.Softmax())

# Here, D_i refers to the input dimensions (the features in the input data), D_h refers
# to the hidden dimensions (the number of nodes in a hidden layer), and D_o refers to
# the output dimensions.
# Using custom modules, it is possible to build an equivalent network architecture, as
# shown here:
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(torch.nn.Module):
    def __init__(self, D_i, D_h, D_o):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(D_i, D_h)
        self.linear2 = torch.nn.Linear(D_h, D_o)

    def forward(self, x):
        z = F.relu(self.linear1(x))
        o = F.softmax(self.linear2(z))

        return o

# As can be seen, an input layer and an output layer are defined inside the initialization
# method of the class. Next, an additional method is defined where the computations
# are performed.


model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 10
batch_size = 100

train_losses, dev_losses, train_acc, dev_acc = [], [], [], []

for e in range(epochs):
    X, y = shuffle(X_train, y_train)
    running_loss = 0
    running_acc = 0
    iterations = 0

    for i in range(0, len(X), batch_size):
        iterations += 1
        b = i + batch_size
        X_batch = torch.tensor(X.iloc[i:b, :].values).float()
        y_batch = torch.tensor(y.iloc[i:b, :].values).float()

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

    with torch.no_grad():
        pred_dev = model(X_dev_torch)
        dev_loss = criterion(pred_dev, y_dev_torch)
        ps_dev = torch.exp(pred_dev)
        top_p, top_class_dev = ps_dev.topk(1, dim=1)
        acc = accuracy_score(y_dev_torch, top_class_dev)

    train_losses.append(running_loss/iterations)
    dev_losses.append(dev_loss)
    train_acc.append(running_acc/iterations)
    dev_acc.apppend(acc)

    print("Epochs: {}/{}..".format(e+1, epochs),
          "Training Loss: {:.3f}..".format(running_loss/iterations),
          "Validation Loss: {:.3f}..".format(dev_loss),
          "Training Accuracy: {:.3f}..".format(running_acc/iterations),
          "Validation Accuracy: {:.3f}".format(acc))


