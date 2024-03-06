# Activity 6.02: Text Generation with LSTM Networks

# 1. Import the required libraries, as follows:
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. Open and read the text from Alice in Wonderland into the notebook. Print an
# extract of the first 50 characters and the total length of the text file
with open('data/alice.txt', 'r', encoding='latin1') as f:
    data = f.read()

print("Extract: ", data[:50])
print("Length: ", len(data))
# Extract:  ALICE was beginning to get very tired of sitting b
# Length:  145178

# 3. Create a variable containing a list of the unduplicated characters in your dataset.
# Then, create a dictionary that maps each character to an integer, where the
# characters will be the keys and the integers will be the values:
chars = list(set(data))
indexer = {char: index for (index, char) in enumerate(chars)}

# 4. Encode each letter of your dataset to its paired integer. Print the first 50 encoded
# characters and the total length of the encoded version of your dataset:
indexed_data = []
for c in data:
    indexed_data.append(indexer[c])

print("Indexed extract: ", indexed_data[:50])
print("Length: ", len(indexed_data))


# Indexed extract:  [59, 31, 5, 10, 35, 2, 53, 47, 38, 2, 56, 20, 46, 8, 21, 21, 8,
# 21, 46, 2, 11, 4, 2, 46, 20, 11, 2, 15, 20, 37, 64, 2, 11, 8, 37, 20, 43, 2, 4, 6,
# 2, 38, 8, 11, 11, 8, 21, 46, 2, 56]
# Length:  145178

# 5. Create a function that takes in a batch and encodes it as a one-hot matrix:
def index2onehot(batch):
    batch_flatten = batch.flatten()
    onehot_flat = np.zeros((batch.shape[0] * batch.shape[1], len(indexer)))
    onehot_flat[range(len(batch_flatten)), batch_flatten] = 1
    onehot = onehot_flat.reshape((batch.shape[0], batch.shape[1], -1))
    return onehot


# This function takes a two-dimensional matrix and flattens it. Next, it creates
# a zero-filled matrix of the shape of the flattened matrix and the length of the
# dictionary containing the alphabet (created in Step 3). Next, it fills the letter that
# corresponds to each character in the batch with ones. Finally, it reshapes the
# matrix so that it's three-dimensional.

# 6. Create a class that defines the architecture of the network. This class should
# contain an additional function that initializes the states of the LSTM layers:
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
        hidden = next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_size).zero_()
        cell = next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_size).zero_()
        states = (hidden, cell)
        return states


# This class contains an __init__ method where the architecture of the network
# is defined, a forward method to determine the flow of the data through the
# layers, and an init_state method to initialize the hidden and cell states
# with zeros.

# 7. Determine the number of batches to be created out of your dataset, bearing in
# mind that each batch should contain 100 sequences, each with a length of 50.
# Next, split the encoded data into 100 sequences:

# Number of sequences per batch
n_seq = 100
seq_length = 50
n_batches = math.floor(len(indexed_data) / n_seq / seq_length)
total_length = n_seq * seq_length * n_batches
x = indexed_data[:total_length]
x = np.array(x).reshape((n_seq, -1))

# 8. Instantiate your model by using 256 as the number of hidden units for a total of
# two recurrent layers:
model = LSTM(len(chars), 256, 2)
print(model)
# LSTM(
#   (lstm): LSTM(70, 256, num_layers=2, batch_first=True)
#   (output): Linear(in_features=256, out_features=70, bias=True)
# )

# If your machine has a GPU available, make sure to allocate the model to the GPU,
# using the following code snippet instead:
# model = LSTM(len(chars), 256, 2).to("cuda")

# 9. Define the loss function and the optimization algorithms. Use the Adam
# optimizer and the cross-entropy loss to do this. Train the network for 20 epochs:
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20

# If your machine has a GPU available, try running the training process for 500
# epochs instead:
# epochs = 500

# 10. In each epoch, the data must be divided into batches with a sequence length
# of 50. This means that each epoch will have 100 batches, each with a sequence
# of 50:
# 在每个时期中，数据必须被划分成长度为50的批次。这意味着每个时期将有100个批次，每个批次有一个长度为50的序列。
losses = []
for e in range(1, epochs+1):
    states = model.init_states(n_seq)
    batch_loss = []

    for b in range(0, x.shape[1], seq_length):
        x_batch = x[:, b:b+seq_length]

        if b == x.shape[1] - seq_length:
            y_batch = x[:, b+1:b+seq_length]
            y_batch = np.hstack((y_batch, indexer["."] * np.ones((y_batch.shape[0], 1))))
        else:
            y_batch = x[:, b+1:b+seq_length+1]
        np_x_onehot = index2onehot(x_batch)
        x_onehot = torch.Tensor(np_x_onehot)
        # x_onehot = torch.Tensor(np_x_onehot).to("cuda")
        y = torch.Tensor(y_batch).view(n_seq * seq_length)
        # y = torch.Tensor(y_batch).view(n_seq * seq_length).to("cuda")

        pred, states = model(x_onehot, states)
        loss = loss_function(pred, y.long())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        # 清除状态
        states = tuple(s.detach() for s in states)

        batch_loss.append(loss.item())

    losses.append(np.mean(batch_loss))

    if e % 2 == 0:
        print("epochs: ", e, "... Loss function: ", losses[-1])

# epoch: 2 ... Loss function: 3.1667490992052802
# epoch: 4 ... Loss function: 3.1473221943296235
# epoch: 6 ... Loss function: 2.897721455014985
# epoch: 8 ... Loss function: 2.567064647016854
# epoch: 10 ... Loss function: 2.4197753791151375
# epoch: 12 ... Loss function: 2.314083896834275
# epoch: 14 ... Loss function: 2.2241266349266313
# epoch: 16 ... Loss function: 2.1459227183769487
# epoch: 18 ... Loss function: 2.0731402758894295
# epoch: 20 ... Loss function: 2.0148646708192497


# 11. Plot the progress of the loss over tim
x_range = range(len(losses))
plt.plot(x_range, losses)
plt.xlabel("epochs")
plt.ylabel("Loss function")
plt.show()
# As we can see, after 20 epochs, the loss function can still be reduced, which is
# why training for more epochs is strongly recommended in order to get a good
# result from the model.

# 12. Feed the following sentence starter into the trained model for it to complete
# the sentence: "So she was considering in her own mind ":
starter = "So she was considering in her own mind "
states = None

# If your machine has a GPU available, allocate the model back to the CPU to
# perform predictions:
# model = model.to("cpu")

# First, a for loop is used to feed the seed into the model so that the memory
# can be generated. Next, the predictions are performed, as can be seen in the
# following snippet:
for ch in starter:
    x = np.array([[indexer[ch]]])
    x = index2onehot(x)
    x = torch.Tensor(x)
    pred, states = model(x, states)

counter = 0
while starter[-1] != "." and counter < 100:
    counter += 1
    x = np.array([[indexer[starter[-1]]]])
    x = index2onehot(x)
    x = torch.Tensor(x)
    pred, states = model(x, states)
    pred = F.softmax(pred, dim=1)
    p, top = pred.topk(10)
    p = p.detach().numpy()[0]
    top = top.numpy()[0]
    index = np.random.choice(top, p=p/p.sum())

    starter += chars[index]

print(starter)

# 这段代码是一个文本生成的示例，它使用已经训练好的 LSTM 模型来生成一段新的文本。具体步骤如下：
#
# 1. 使用 for 循环将种子文本传入模型，以生成内部的记忆（隐藏状态）。
# 2. 在生成种子文本的基础上，循环执行以下步骤：
#    - 将最后一个字符转换为对应的 one-hot 编码。
#    - 将 one-hot 编码转换为 PyTorch 张量。
#    - 将张量传入模型中进行预测，得到预测结果和更新后的状态。
#    - 对预测结果进行 softmax 处理，以获取每个字符的概率分布。
#    - 根据概率分布从前 10 个字符中随机选择一个字符作为下一个字符。
#    - 将选择的字符添加到生成的文本末尾。
# 3. 循环直到生成的文本以句号结尾或者达到最大迭代次数。
#
# 这段代码演示了如何使用训练好的 LSTM 模型来生成类似风格的文本。
# 通过在模型中不断传入先前生成的文本和状态，模型能够逐步生成新的文本，其中包含了种子文本的风格和主题。
