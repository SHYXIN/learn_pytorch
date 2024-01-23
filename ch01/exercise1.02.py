import torch
torch.manual_seed(0)

import torch.nn as nn

input_units = 10
output_units = 1

model = nn.Sequential(nn.Linear(input_units, output_units),
                      nn.Sigmoid())

print(model)

loss_funct = nn.MSELoss()
print(loss_funct)