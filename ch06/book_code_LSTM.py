from torch import nn

class LSTM(nn.Module):
    def __init__(self, char_length, hidden_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(char_length, hidden_size, n_layers,
                            batch_first=True)
        self.output = nn.Linear(hidden_size, char_length)

    def forward(self, x, states):
        out, states = self.lstm(x, states)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.output(out)

        return out, states

    def init_states(self, batch_size):
        hidden = next(self.parameters()).data.new(self.n_layers, batch_size,
                                                  self.hidden_size).zero_()
        cell = next(self.parameters()).data.new(self.n_layers, batch_size,
                                                self.hidden_size).zeros_()
        states = (hidden, cell)
        return states

# Training the Model
# Once the loss function and the optimization algorithm have been defined, it is time
# to train the model. This is achieved by following a very similar approach to the
# one that is used for other neural network architectures, as shown in the following
# code snippet:

# Step 1: for through epochs
for e in range(1, epochs+1):
    # Step 2: Memory initialized
    states = model.init_state(n_seq)

    # Step 3: for loop to split data in batches:
    for b in range(0, x.shape[1], seq_length):
        x_batch = x[:,b:b+seq_length]

        if b == x.shape[1] - seq_length:
            y_batch = x[:,b+1:b_seq_length]
            y_batch = np.hstack((y_batch, indexer["."] * np.ones((y_batch.shape[0]))))
        else:
            y_batch = x[:,b+1:b+seq_length+1]
        """
        Step 4 input data is converted to one-hot matrix.
        Inputs and targets are converted to tensors.
        """
        x_onehot = torch.Tensor(index2onehot(x_batch))

        """
        Step 5: get a predction and preform the backward propagation
        """
        pred, states = model(x_onehot, states)
        loss = loss_function(pred, y.long())
        optimizer.zero_grad()
        loss.backward(return_graph=True)
        optimizer.step()