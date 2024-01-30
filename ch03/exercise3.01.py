
# Using the theory explained previously, we will define a model's architecture using the
# custom module's syntax:
# 1. Open a Jupyter Notebook and import the required libraries:
import torch
import torch.nn as nn
import torch.nn.functional as F

# 2. Define the necessary variables for the input, hidden, and output dimensions. Set
# them to 10, 5, and 2, respectively:
D_i = 10
D_h = 5
D_o = 2

# 3. Using PyTorch's custom modules, create a class called Classifier and
# define the model's architecture so that it has two linear layers—the first one
# followed by a ReLU activation function, and the second one by a Softmax
# activation function:

class Classifier(torch.nn.Module):
    def __init__(self, D_i, D_h, D_o):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(D_i, D_h)
        self.linear2 = torch.nn.Linear(D_h, D_o)

    def forward(self, x):
        z = F.relu(self.linear1(x))
        o = F.softmax(self.linear2(z))

        return o

# 4. Instantiate the class and feed it with the three variables we created in Step 2.
# Print the model:
model = Classifier(D_i, D_h, D_o)
print(model)

# Classifier(
#   (linear1): Linear(in_features=10, out_features=5, bias=True)
#   (linear2): Linear(in_features=5, out_features=2, bias=True)
# )
# With that, you have successfully built a neural network architecture using PyTorch's
# custom modules. Now, you can move on and learn about the process of training a
# deep learning model.