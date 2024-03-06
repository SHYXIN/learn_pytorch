# Exercise 6.02: Preprocessing the Input Data and Creating a One-Hot Matrix
# In this exercise, you will preprocess a text snippet, which will then be converted into a
# one-hot matrix. Follow these steps to complete this exercise:
# 1. Import NumPy:
import numpy as np

# 2. Create a variable named text, which will contain the text sample
# "Hello World!":
text = "Hello World!"

# 3. Create a dictionary by mapping each letter to a number:
chars = list(set(text))
indexer = {char: index for (index, char) in enumerate(chars)}
print(indexer)

# Running the preceding code will result in the following output:
# {'o': 0, 'H': 1, ' ': 2, 'l': 3, 'r': 4, 'd': 5, 'e': 6, 'W': 7}

# 4. Encode your text sample with the numbers we defined in the previous step:
encoded = []
for c in text:
    encoded.append(indexer[c])
# 5. Convert the encoded variable into a NumPy array and reshape it so that the
# sentence is divided into two sequences of the same size:
encoded = np.array(encoded).reshape((2, -1))
print(encoded)


# [[7 3 8 8 2 1]
#  [0 2 6 8 5 4]]

# 6. Define a function that takes an array of numbers and creates a one-hot matrix:
def index2onehot(batch):
    batch_flatten = batch.flatten()
    onehot_flat = np.zeros((batch.shape[0] * batch.shape[1], len(indexer)))
    onehot_flat[range(len(batch_flatten)), batch_flatten] = 1
    onehot = onehot_flat.reshape((batch.shape[0], batch.shape[1], -1))
    return onehot


# 7. Convert the encoded array into a one-hot matrix by passing it through the
# previously defined function:
one_hot = index2onehot(encoded)
print(one_hot)
# [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]]
#
#  [[0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]]]
# You have successfully converted some sample text into a one-hot matrix.
