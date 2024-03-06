import pandas as pd
from string import punctuation

test = pd.Series(['Hey! This is example #1.',
                  'Hey! This is example #2.',
                  'Hey! This is example #3.'])

for i in punctuation:
    test = test.str.replace(i, "")

print(test)

import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, n_layers)
        self.output = nn.Linear(hidden_size)

    def forward(self, x, states):
        out = self.embedding(x)
        out, states = self.lstm(out, states)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.output(out)

        return out, states


