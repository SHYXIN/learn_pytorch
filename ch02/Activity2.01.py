

# 1. Import the required libraries.
import pandas as pd
# 2. Using pandas, load the .csv file.
data = pd.read_csv("YearPredictionMSD.csv")
print(data.head())
# 3. Verify whether any qualitative data is present in the dataset.
cols = data.columns
num_cols = data._get_numeric_data()
cate_cols = list(set(cols) - set(num_cols))
print(cate_cols)
# 4. Check for missing values.
# You can also add an additional sum() function to get the sum of missing values
# in the entire dataset, without discriminating by column.
print(data.isnull().sum())

# 5. Check for outliers.
outliers = {}
for i in range(data.shape[1]):
    col_mean = data[data.columns[i]].mean()
    col_std = data[data.columns[i]].std()
    min_t = col_mean - 3 * col_std
    max_t = col_mean + 3 * col_std
    count = 0
    for j in data[data.columns[i]]:
        if j > max_t or j < min_t:
            count += 1
    percentage = count / data.shape[0]
    outliers[data.columns[i]] = '%.3f' % percentage

print(outliers)

# 6. Separate the features from the target data.
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]


# 7. Rescale the data using the standardization methodology.
X = (X - X.mean()) / X.std()
print(X.head())

from sklearn.model_selection import train_test_split

X_shuffle = X.sample(frac=1, random_state=0)
Y_shuffle = Y.sample(frac=1, random_state=0)

x_new, x_test, y_new, y_test = train_test_split(X_shuffle, Y_shuffle, test_size=0.2, random_state=0)
dev_per = x_test.shape[0] / x_new.shape[0]
x_train, x_dev, y_train, y_dev = train_test_split(x_new, y_new, test_size=dev_per, random_state=0)

print(x_train.shape, y_train.shape)
print(x_dev.shape, y_dev.shape)
print(x_test.shape, y_test.shape)
# (30000, 90) (30000,)
# (10000, 90) (10000,)
# (10000, 90) (10000,)

# Activity2.02 Developing a Deep Learning Solution for a Regression Problem
# In this activity, we will create and train a neural network to solve the regression
# problem we mentioned in the previous activity. Let's look at the scenario.

# You continue to work at the music record company and, after seeing the great job
# you did preparing the dataset, your boss has trusted you with the task of defining the
# network's architecture, as well as training it with the prepared dataset. Perform the
# following steps to complete this activity:

# 1. Import the required libraries.
import torch
import torch.nn as nn

torch.manual_seed(0)

# 2. Split the features from the targets for all three sets of data that we created in the
# previous activity. Convert the DataFrames into tensors.

x_train = torch.tensor(x_train.values).float()
y_train = torch.tensor(y_train.values).float()

x_dev = torch.tensor(x_dev.values).float()
y_dev = torch.tensor(y_dev.values).float()

x_test = torch.tensor(x_test.values).float()
y_test = torch.tensor(y_test.values).float()



# 3. Define the architecture of the network. Feel free to try different combinations of
# the number of layers and the number of units per layer.
model = nn.Sequential(nn.Linear(x_train.shape[1], 10),
                      nn.ReLU(),

                      nn.Linear(10, 7),
                      nn.ReLU(),

                      nn.Linear(7, 5),
                      nn.ReLU(),

                      nn.Linear(5, 1)
                      )
# import tensorflow as tf
# from tensorflow.keras import layers
#
# # 定义模型
# model = tf.keras.Sequential([
#     layers.Dense(10, input_shape=(x_train.shape[1],), activation='relu'),
#     layers.Dense(7, activation='relu'),
#     layers.Dense(5, activation='relu'),
#     layers.Dense(1)
# ])
#
# # 打印模型概要
# model.summary()


# 4. Define the loss function and the optimizer algorithm.
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. Use a for loop to train the network for 3,000 iteration steps.
for i in range(3000):
    y_pred = model(x_train).squeeze()
    loss = loss_function(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 250 == 0:
        print(i, loss.item())
# 0 3994019.0
# 250 201386.703125
# 500 14111.64453125
# 750 1335.8360595703125
# 1000 395.67578125
# 1250 194.95230102539062
# 1500 133.74514770507812
# 1750 110.15707397460938
# 2000 100.00774383544922
# 2250 95.13729095458984
# 2500 92.48408508300781
# 2750 90.95240783691406

# 6. Test your model by performing a prediction on the first instance of the test set
# and comparing it with the ground truth
pred = model(x_test[0])
print("Ground truth:", y_test[0].item(), "Prediction:", pred.item())

# Ground truth: 1995.0 Prediction: 1998.01806640625


