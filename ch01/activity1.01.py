# https://archive.ics.uci.edu/dataset/479/somerville+happiness+survey

# https://github.com/PacktWorkshops/The-Deep-Learning-with-PyTorch-Workshop/blob/master/Chapter01/Activity1.01/SomervilleHappinessSurvey2015.csv

# 1. Import the required libraries, including pandas for reading a CSV file.
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. Read the CSV file containing the dataset
dataset = pd.read_csv('SomervilleHappinessSurvey2015.csv')
print(dataset.head())

# 3. Separate the input features from the target. Note that the target is located in the
# first column of the CSV file. Next, convert the values into tensors, making sure
# the values are converted into floats.
x = torch.tensor(dataset.iloc[:, 1:].values).float()
y = torch.tensor(dataset.iloc[:, :1].values).float()
print(x[:5])
print(y[:5])

# 4. Define the architecture of the model and store it in a variable named model.
# Remember to create a single-layer model.
model = nn.Sequential(nn.Linear(6, 1), nn.Sigmoid())

# 5. Define the loss function to be used. In this case, use the MSE loss function.
loss_funct = nn.MSELoss()

# 6. Define the optimizer of your model. In this case, use the Adam optimizer and a
# learning rate of 0.01.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 7. Run the optimization for 100 iterations, saving the loss value for each iteration.
# Print the loss value every 10 iterations.
losses = []
for i in range(100):
    y_pred = model(x)
    loss = loss_funct(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(i, loss.item())

# 8. Make a line plot to display the loss value for each iteration step.
plt.plot(range(100), losses)
plt.show()