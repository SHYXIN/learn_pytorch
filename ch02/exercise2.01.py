# 2. Import the pandas library:
import pandas as pd

# 3. Use pandas to read the CSV file containing the dataset we downloaded from the
# UC Irvine Machine Learning Repository site.
# Next, drop the column named date as we do not want to consider it for the
# following exercises:
data = pd.read_csv("energydata_complete.csv")
data = data.drop(columns=["date"])
print(data.head())
#    Appliances  lights     T1  ...  Tdewpoint        rv1        rv2
# 0          60      30  19.89  ...        5.3  13.275433  13.275433
# 1          60      30  19.89  ...        5.2  18.606195  18.606195
# 2          50      30  19.89  ...        5.1  28.642668  28.642668
# 3          50      40  19.89  ...        5.0  45.410389  45.410389
# 4          60      40  19.89  ...        4.9  10.084097  10.084097
# 4. Check for categorical features in your dataset:
cols = data.columns

num_columns = data._get_numeric_data().columns
cate_columns = list(set(cols) - set(num_columns))
print(cate_columns)
# []
# The first line generates a list of all the columns in your dataset. Next, the
# columns that contain numeric values are stored in a variable as well. Finally, by
# subtracting the numeric columns from the entire list of columns, it is possible to
# obtain those that are not numeric.
# The resulting list is empty, which indicates that there are no categorical features
# to deal with.

# 5. Use Python's isnull() and sum() functions to find out whether there are any
# missing values in each column of the dataset:
print(data.isnull().sum())
# This command counts the number of null values in each column. For the dataset
# in use, there should not be any missing values, as can be seen here:

# Appliances     0
# lights         0
# T1             0
# RH_1           0
# T2             0
# RH_2           0
# T3             0
# RH_3           0
# T4             0
# RH_4           0
# T5             0
# RH_5           0
# T6             0
# RH_6           0
# T7             0
# RH_7           0
# T8             0
# RH_8           0
# T9             0
# RH_9           0
# T_out          0
# Press_mm_hg    0
# RH_out         0
# Windspeed      0
# Visibility     0
# Tdewpoint      0
# rv1            0
# rv2            0

# 6. Use three standard deviations as the measure to detect any outliers for all the
# features in the dataset:
outliers = {}
for i in range(data.shape[1]):
    min_t = data[data.columns[i]].mean() - 3 * data[data.columns[i]].std()
    max_t = data[data.columns[i]].mean() + 3 * data[data.columns[i]].std()

    count = 0
    for j in data[data.columns[i]]:
        if j < min_t or j > max_t:
            count +=1
    percentage = count / data.shape[0]
    outliers[data.columns[i]] = "%.3f" % percentage
print(outliers)
# The preceding code snippet performs a for loop through the columns in the
# dataset in order to evaluate the presence of outliers in each of them. It continues
# to calculate the minimum and maximum thresholds so that it can count the
# number of instances that fall outside the range between the thresholds.
# Finally, it calculates the percentage of outliers (that is, the number of outliers
# divided by the total number of instances) in order to output a dictionary that
# displays this percentage for each column.

# By printing the resulting dictionary (outliers), it is possible to display a list of
# all the features (columns) in the dataset, along with the percentage of outliers.
# According to the result, it is possible to conclude that there is no need to deal
# with the outlier values, considering that they account for less than 5% of the
# data, as can be seen in the following screenshot

# Exercise 2.03
# normalization
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]

X = (X - X.min()) / (X.max() - X.min())
print(X.head())

# # standard
# X = data.iloc[:, 1:]
# Y = data.iloc[:, 0]
#
# X = (X - X.mean()) / X.std()


# Exercise2.04
# n this exercise, we will split the dataset from the previous exercise into three subsets.
# For the purpose of learning, we will explore two different approaches. First, the
# dataset will be split using indexing. Next, scikit-learn's train_test_split()
# function will be used for the same purpose, thereby achieving the same result with
# both approaches. Perform the following steps to complete this exercise:
print(X.shape)
# (19735, 27)
# The output from this operation should be (19735, 27). This means that it is
# possible to use a split ratio of 60:20:20 for the training, validation, and test sets.

# 2. Get the value that you will use as the upper bound of the training and validation
# sets. This will be used to split the dataset using indexing:

train_end = int(len(X) * 0.6)
dev_end = int(len(X) * 0.8)
# The preceding code determines the index of the instances that will be used to
# divide the dataset through slicing.

# 3. Shuffle the dataset:
X_shuffle = X.sample(frac=1, random_state=0)
Y_shuffle = Y.sample(frac=1, random_state=0)

# 4. Use indexing to split the shuffled dataset into the three sets for both the features
# and the target data
x_train = X_shuffle.iloc[:train_end, :]
y_train = Y_shuffle.iloc[:train_end]
x_dev = X_shuffle.iloc[train_end:dev_end, :]
y_dev = Y_shuffle.iloc[train_end:dev_end]
x_test = X_shuffle.iloc[dev_end:]
y_test = Y_shuffle.iloc[dev_end:]
# 5. Print the shapes of all three sets:
print(x_train.shape, y_train.shape)
print(x_dev.shape, y_dev.shape)
print(x_test.shape, y_test.shape)
# (11841, 27) (11841,)
# (3947, 27) (3947,)
# (3947, 27) (3947,)
# 6. Import the train_test_split() function from scikit-learn's
# model_selection module:
from sklearn.model_selection import train_test_split
x_new, x_test_2, y_new, y_test_2 = train_test_split(X_shuffle, Y_shuffle,test_size=0.2,random_state=0)

dev_per = x_test_2.shape[0] / x_new.shape[0]
print(dev_per)
x_train_2, x_dev_2, y_train_2, y_dev_2 = train_test_split(x_new, y_new, test_size=dev_per,random_state=0)
# The first line of code performs an initial split. The function takes the following
# as arguments:
# X_shuffle, Y_shuffle: The datasets to be split, that is, the features dataset,
# as well as the target dataset (also known as X and Y)
# test_size: The percentage of instances to be contained in the testing set
# random_state: Used to ensure the reproducibility of the results
# The result from this line of code is the division of each of the datasets (X and Y)
# into two subsets.
# To create an additional set (the validation set), we will perform a second split.
# The second line of the preceding code is in charge of determining the test_
# size to be used for the second split so that both the testing and validation sets
# have the same shape.
# Finally, the last line of code performs the second split using the value that was
# calculated previously as the test_size.

# 8. Print the shape of all three sets
print(x_train_2.shape, y_train_2.shape)
print(x_dev_2.shape, y_dev_2.shape)
print(x_test_2.shape, y_test_2.shape)

# As we can see, the resulting sets from both approaches have the same shapes.
# Using one approach or the other is a matter of preference.


# Exercise 2.05: Building a Deep Neural Network Using PyTorch
# In this exercise, we will use the PyTorch library to define the architecture of a
# deep neural network of four layers, which will then be trained with the dataset we
# prepared in the previous exercises. Perform the following steps to do so:

# 1. Import the PyTorch library, called torch, as well as the nn module
# from PyTorch:
import torch
import torch.nn as nn

torch.manual_seed(0)
# 2. Separate the feature columns from the target for each of the sets we created in
# the previous exercise. Additionally, convert the final DataFrames into tensors
x_train = torch.tensor(x_train.values).float()
y_train = torch.tensor(y_train.values).float()

x_dev = torch.tensor(x_dev.values).float()
y_dev = torch.tensor(y_dev.values).float()

x_test = torch.tensor(x_test.values).float()
y_test = torch.tensor(y_test.values).float()

# 3. Define the network architecture using the sequential() container. Make sure
# to create a four-layer network. Use ReLU activation functions for the first three
# layers and leave the last layer without an activation function, considering the fact
# that we are dealing with a regression problem.

# The number of units for each layer should be 100, 50, 25, and 1:
model = nn.Sequential(nn.Linear(x_train.shape[1], 100),
                      nn.ReLU(),
                      nn.Linear(100, 50),
                      nn.ReLU(),
                      nn.Linear(50, 25),
                      nn.ReLU(),
                      nn.Linear(25, 1)
                      )

# 4. Define the loss function as the MSE:
loss_function = torch.nn.MSELoss()

# 5. Define the optimizer algorithm as the Adam optimizer:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. Use a for loop to train the network over the training data for 1,000
# iteration steps:
for i in range(1000):
    y_pred = model(x_train).squeeze()
    loss = loss_function(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(i, loss.item())

# 0 20351.2734375
# 100 10840.431640625
# 200 10686.6240234375
# 300 10449.0927734375
# 400 10123.9736328125
# 500 9767.0244140625
# 600 9480.7265625
# 700 9205.9296875
# 800 8967.60546875
# 900 8721.388671875

# The squeeze() function is used to remove the additional dimension of
# y_pred, which is converted from being of size [3000,1] to [3000].
# This is crucial considering that y_train is one-dimensional and both
# tensors need to have the same dimensions to be fed to the loss function.


# 7. To test the model, perform a prediction on the first instance of the testing set
# and compare it with the ground truth (target value):
pred = model(x_test[0])
print("Ground truth:", y_test[0].item(),
      "Prediction:", pred.item())
# The output should look similar to the following

# Ground truth: 60.0 Prediction: 69.23514556884766

# As you can see, the ground truth value (60) is fairly close to the predicted one (69.58).
# You have successfully created and trained a deep neural network to solve a
# regression problem.





