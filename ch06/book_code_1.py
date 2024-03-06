import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, self.hiddent_size)
        out = self.output(out)
        return out, hidden

# Here, the recurrent layer must be defined as taking arguments for the number of
# expected features in the input (input_size); the number of features in the hidden
# state, which is defined by the user (hidden_size); and the number of recurrent
# layers (num_layers).

for i in range(1, epochs+1):

    hidden = None

    for inputs, targets in batchs:
        pred, hidden = model(inputs, hidden)

        loss = loss_function(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

