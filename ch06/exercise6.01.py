# Exercise 6.01: Creating the Input and Target Variables for a Sequenced Data
# Problem

# 1. Import the following libraries:
import pandas as pd
import numpy as np
import torch

# 2. Create a Pandas DataFrame that's 10 x 5 in size, filled with random numbers
# ranging from 0 to 100. Name the five columns as follows: ["Week1",
# "Week2", "Week3", "Week4", "Week5"].
# Make sure to set the random seed to 0 to be able to reproduce the results
# shown in this book:
np.random.seed(0)
data = pd.DataFrame(np.random.randint(0, 100, size=(10, 5)),
                    columns=['Week1', 'Week2', 'Week3', 'Week4', 'Week5'])

print(data)
#    Week1  Week2  Week3  Week4  Week5
# 0     44     47     64     67     67
# 1      9     83     21     36     87
# 2     70     88     88     12     58
# 3     65     39     87     46     88
# 4     81     37     25     77     72
# 5      9     20     80     69     79
# 6     47     64     82     99     88
# 7     49     29     19     19     14
# 8     39     32     65      9     57
# 9     32     31     74     23     35

# 3. Create an input and a target variable, considering that the input variable should
# contain all the values of all the instances, except the last column of data. The
# target variable should contain all the values of all the instances, except the
# first column:
inputs = data.iloc[:, :-1]
print(inputs)
#    Week1  Week2  Week3  Week4
# 0     44     47     64     67
# 1      9     83     21     36
# 2     70     88     88     12
# 3     65     39     87     46
# 4     81     37     25     77
# 5      9     20     80     69
# 6     47     64     82     99
# 7     49     29     19     19
# 8     39     32     65      9
# 9     32     31     74     23

targets = inputs.shift(-1, axis="columns", fill_value=data.iloc[:, -1:])
print(targets)
#  Week1 Week2 Week3 Week4
# 0    47    64    67    67
# 1    83    21    36    87
# 2    88    88    12    58
# 3    39    87    46    88
# 4    37    25    77    72
# 5    20    80    69    79
# 6    64    82    99    88
# 7    29    19    19    14
# 8    32    65     9    57
# 9    31    74    23    35

my_data = pd.DataFrame(np.arange(50).reshape(10,5), columns=[f'Week{i}' for i in range(1, 6)])
print(my_data)
my_inputs = my_data.iloc[:, :-1]
print(my_inputs)
my_target1 = my_inputs.shift(-1, axis="columns")
print(my_target1)
my_target = my_inputs.shift(-1, axis="columns", fill_value=my_data.iloc[:, -1:])
print(my_target)

#   Week1  Week2  Week3  Week4  Week5
# 0      0      1      2      3      4
# 1      5      6      7      8      9
# 2     10     11     12     13     14
# 3     15     16     17     18     19
# 4     20     21     22     23     24
# 5     25     26     27     28     29
# 6     30     31     32     33     34
# 7     35     36     37     38     39
# 8     40     41     42     43     44
# 9     45     46     47     48     49
#    Week1  Week2  Week3  Week4
# 0      0      1      2      3
# 1      5      6      7      8
# 2     10     11     12     13
# 3     15     16     17     18
# 4     20     21     22     23
# 5     25     26     27     28
# 6     30     31     32     33
# 7     35     36     37     38
# 8     40     41     42     43
# 9     45     46     47     48
#    Week1  Week2  Week3  Week4
# 0    1.0    2.0    3.0    NaN
# 1    6.0    7.0    8.0    NaN
# 2   11.0   12.0   13.0    NaN
# 3   16.0   17.0   18.0    NaN
# 4   21.0   22.0   23.0    NaN
# 5   26.0   27.0   28.0    NaN
# 6   31.0   32.0   33.0    NaN
# 7   36.0   37.0   38.0    NaN
# 8   41.0   42.0   43.0    NaN
# 9   46.0   47.0   48.0    NaN
#   Week1 Week2 Week3 Week4
# 0     1     2     3     4
# 1     6     7     8     9
# 2    11    12    13    14
# 3    16    17    18    19
# 4    21    22    23    24
# 5    26    27    28    29
# 6    31    32    33    34
# 7    36    37    38    39
# 8    41    42    43    44
# 9    46    47    48    49