# Activity 6.01: Using a Simple RNN for a Time Series Prediction

# 1. Import the required libraries, as follows:
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn, optim

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. Load the dataset and then slice it so that it contains all the rows but only the
# columns from index 1 to 52:
data = pd.read_csv("data/Sales_Transactions_Dataset_Weekly.csv")
data = data.iloc[:, 1:53]
print(data.head())
#    W0  W1  W2  W3  W4  W5  W6  W7  ...  W44  W45  W46  W47  W48  W49  W50  W51
# 0  11  12  10   8  13  12  14  21  ...    8   10   12    3    7    6    5   10
# 1   7   6   3   2   7   1   6   3  ...    5    1    1    4    5    1    6    0
# 2   7  11   8   9  10   8   7  13  ...    5    5    7    8   14    8    8    7
# 3  12   8  13   5   9   6   9  13  ...    3    4    6    8   14    8    7    8
# 4   8   5  13  11   6   7   9  14  ...    7   12    6    6    5   11    8    9

# 3. Plot the weekly sales transactions of five randomly chosen products from the
# entire dataset. Use a random seed of 0 when performing random sampling in
# order to achieve the same results as in the current activity:
plot_data = data.sample(5, random_state=0)
x = range(1, 53)
plt.figure(figsize=(10, 5))

for i, row in plot_data.iterrows():
    plt.plot(x, row)

plt.legend(plot_data.index)
plt.xlabel("Weeks")
plt.ylabel("Sales transaction per product")
plt.show()

# 4. Create the inputs and targets variables that will be fed to the network
# to create the model. These variables should be of the same shape and be
# converted into PyTorch tensors.
# The inputs variable should contain the data for all the products for all the
# weeks except the last week, since the idea of the model is to predict this
# final week.
# The targets variable should be one step ahead of the inputs variable; that is,
# the first value of the targets variable should be the second one of the inputs
# variable and so on until the last value of the targets variable (which should be
# the last week that was left outside of the inputs variable):

data_train = data.iloc[:, :-1]
inputs = torch.Tensor(data_train.values).unsqueeze(1)
# 将训练数据转换为 PyTorch 张量，并在第二个维度上添加一个维度，以适应模型的输入要求。
# 这样处理后，inputs 的形状将变为 (样本数量, 1, 特征数量)。
targets = data_train.shift(-1, axis="columns", fill_value=data.iloc[:, -1]).astype(dtype="float32")

targets = torch.Tensor(targets.values)


# 5. Create a class containing the architecture of the network. Note that the output
# size of the fully connected layer should be 1:
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, self.hidden_size)
        out = self.output(out)
        return out, hidden


# As in the previous activities, the class contains an __init__ method, along with
# the network architecture, and a forward method that determines the flow of
# the information through the layers.

# 6. Instantiate the class function containing the model. Feed the input size, the
# number of neurons in each recurrent layer (10), and the number of recurrent
# layers (1):
model = RNN(data_train.shape[1], 10, 1)
print(model)

# Running the preceding code displays the following output:

# RNN(
#   (rnn): RNN(51, 10, batch_first=True)
#   (output): Linear(in_features=10, out_features=1, bias=True)
# )

# 7. Define a loss function, an optimization algorithm, and the number of epochs to
# train the network. Use the MSE loss function, the Adam optimizer, and 10,000
# epochs to do this:
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10000

# 8. Use a for loop to perform the training process by going through all the epochs.
# In each epoch, a prediction must be made, along with the subsequent calculation
# of the loss function and the optimization of the parameters of the network. Save
# the loss of each of the epochs:

# Note
# Considering that no batches were used to go through the dataset, the
# hidden variable is not actually being passed from batch to batch (rather,
# the hidden state is used while each element of the sequence is being
# processed), but it was left here for clarity
losses = []
for i in range(1, epochs+1):
    hidden = None
    pred, hidden = model(inputs, hidden)
    target = targets[:, -1].unsqueeze(1)

    loss = loss_function(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if i % 1000 == 0:
        print("epoch: ", i, "=... Loss function: ", losses[-1])

# epoch: 1000 ... Loss function: 58.48879623413086
# epoch: 2000 ... Loss function: 24.934917449951172
# epoch: 3000 ... Loss function: 13.247632026672363
# epoch: 4000 ... Loss function: 9.884735107421875
# epoch: 5000 ... Loss function: 8.778228759765625
# epoch: 6000 ... Loss function: 8.025042533874512
# epoch: 7000 ... Loss function: 7.622503757476807
# epoch: 8000 ... Loss function: 7.4796295166015625
# epoch: 9000 ... Loss function: 7.351718902587891
# epoch: 10000 ... Loss function: 7.311776161193848
x_range = range(len(losses))
plt.plot(x_range, losses)
plt.xlabel("epochs")
plt.ylabel("Loss function")
plt.show()

# 10. Using a scatter plot, display the predictions that were obtained in the last
# epoch of the training process against the ground truth values (that is, the sales
# transactions of the last week):
x_range = range(len(data))
target = data.iloc[:, -1].values.reshape(len(data), 1)

plt.figure(figsize=(15, 5))
plt.scatter(x_range[:20], target[:20])
plt.scatter(x_range[:20], pred.detach().numpy()[:20])
plt.legend(["Ground truth", "Prediction"])
plt.xlabel("Product")
plt.ylabel("Sales Transactions")
plt.xticks(range(0, 20))
plt.show()

