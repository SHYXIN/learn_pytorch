
text = "this is a test text!"
chars = list(set(text))
indexer = {char: index for (index, char) in enumerate(chars)}

indexed_data = []
for c in text:
    indexed_data.append(indexer[c])
print(indexer)
print(indexed_data)


indexed_data = list(range(1, 21))
import numpy as np
x = np.array(indexed_data).reshape((2, -1))
print(x)

for b in range(0, x.shape[1], 5):
    batch = x[:, b:b+5]
    print(batch)

# [[ 1  2  3  4  5  6  7  8  9 10]
#  [11 12 13 14 15 16 17 18 19 20]]

# [[ 1  2  3]
#  [11 12 13]]
# [[ 4  5  6]
#  [14 15 16]]
# [[ 7  8  9]
#  [17 18 19]]
# [[10]
#  [20]]

batch = np.array([[2, 4, 7, 6, 5],
                  [2, 1, 6, 2, 5]])

batch_flatten = batch.flatten()
onehot_hat = np.zeros((batch.shape[0] * batch.shape[1], len(indexer)))
onehot_hat[range(len(batch_flatten)), batch_flatten] = 1
onehot = onehot_hat.reshape((batch.shape[0], batch.shape[1], -1))
print(onehot.shape)
print(onehot)
# (2, 5, 9)
# [[[0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]]
#
#  [[0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]]]