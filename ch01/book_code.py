import torch

a = torch.tensor([5.0, 3.0], requires_grad=True)
b = torch.tensor([1.0, 4.0])
ab = ((a + b) ** 2).sum()
ab.backward()

print(a.grad.data)

# print(b.grad.data)
# AttributeError: 'NoneType' object has no attribute 'data'

import torch.nn as nn
model = nn.Sequential(nn.Linear(input_units, hidden_units),
                      nn.ReLU(),
                      nn.Linear(hidden_units, output_units),
                      nn.Sigmoid())
loss_funct = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for i in range(100):
    # Call to the model to perform a prediction
    y_pred = model(x)

    # Calculation of loss function based on y_pred and y
    loss = loss_funct(y_pred, y)

    # Zero the gradients so that previous ones don't accumulate
    optimizer.zero_grad()

    # Calculate the gradients of the loss function
    loss.backward()

    """
    Call the optimizer to perform an update of the parameters
    """
    optimizer.step()
