import torch
torch.manual_seed(0)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch.nn as nn

input_units = 10
output_units = 1

model = nn.Sequential(nn.Linear(input_units, output_units), nn.Sigmoid())

print(model)

loss_funct = nn.MSELoss()
print(loss_funct)


# 1. Import torch, the optim package from PyTorch, and matplotlib:
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# 2、Create dummy input data (x) of random values and dummy target data (y) that
# only contains zeros and ones. Tensor x should have a size of (20,10), while the
# size of y should be (20,1)
x = torch.randn(20, 10)
y = torch.randint(0, 2, (20,1)).type(torch.FloatTensor)

# 3. Define the optimization algorithm as the Adam optimizer. Set the learning rate
# equal to 0.01:
optimizer = optim.Adam(model.parameters())

# 4. Run the optimization for 20 iterations, saving the value of the loss in a variable.
# Every five iterations, print the loss value:
losses = []
for i in range(20):
    y_pred = model(x)
    loss = loss_funct(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 5 == 0:
        print(i, loss.item())

# 0 0.25244325399398804
# 5 0.2505386769771576
# 10 0.24865761399269104
# 15 0.24680247902870178
# The preceding output displays the epoch number, as well as the value for the
# loss function, which, as can be seen, is decreasing. This means that the training
# process is minimizing the loss function, which means that the model is able to
# understand the relationship between the input features and the target.

# 5、Make a line plot to display the value of the loss function in each epoch:
plt.plot(list(range(0, 20)), losses)
plt.show()
