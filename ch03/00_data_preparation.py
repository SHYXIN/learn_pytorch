# The following is a snippet that's used to read the Excel file using pandas and
# print out the head of the dataset:
import pandas as pd

data = pd.read_excel("default of credit card clients.xls", skiprows=1)

print(data.head())
#    ID  LIMIT_BAL  SEX  ...  PAY_AMT5  PAY_AMT6  default payment next month
# 0   1      20000    2  ...         0         0                           1
# 1   2     120000    2  ...         0      2000                           1
# 2   3      90000    2  ...      1000      5000                           0
# 3   4      50000    2  ...      1069      1000                           0
# 4   5      50000    1  ...       689       679                           0

# The shape of the dataset is 30,000 rows and 25 columns, which can be obtained
# using the following line of code:
print("rows: ", data.shape[0], "columns: ", data.shape[1])
# Remove irrelevant features: By performing analysis on each of the features,
# it is possible to determine that two of the features should be removed from the
# dataset as they are irrelevant to the purpose of the study:
data_clean = data.drop(columns=['ID', 'SEX'])
print(data_clean.head())

#    LIMIT_BAL  EDUCATION  ...  PAY_AMT6  default payment next month
# 0      20000          2  ...         0                           1
# 1     120000          2  ...      2000                           1
# 2      90000          2  ...      5000                           0
# 3      50000          2  ...      1000                           0
# 4      50000          2  ...       679                           0
#
# [5 rows x 23 columns]
# The resulting dataset should contain 23 columns instead of the original 25,
# as shown in the following screenshot:

# Check for missing values: Next, it is time to check whether the dataset is
# missing any values, and, if so, calculate the percentage of how much they
# represent each feature, which can be done using the following lines of code:
total = data_clean.isnull().sum()
percent = (data_clean.isnull().sum() / data_clean.isnull().count()*100)
print(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose())
#         LIMIT_BAL  EDUCATION  ...  PAY_AMT6  default payment next month
# Total          0.0        0.0  ...       0.0                         0.0
# Percent        0.0        0.0  ...       0.0                         0.0
#
# [2 rows x 23 columns]
# From these results, it is possible to say that the dataset is not missing any values,
# so no further action is required here.

# • Check for outliers: As we mentioned in Chapter 2, Building Blocks of Neural
# Networks, there are several ways to check for outliers. However, in this book, we
# will stick to the standard deviation methodology, where values that are three
# standard deviations away from the mean will be considered outliers. Using the
# following code, it is possible to identify outliers from each feature and calculate
# the proportion they represent against the entire set of values:
outliers = {}

for i in range(data_clean.shape[1]):
    min_t = data_clean[data_clean.columns[i]].mean() - 3 * data_clean[data_clean.columns[i]].std()
    max_t = data_clean[data_clean.columns[i]].mean() + 3 * data_clean[data_clean.columns[i]].std()
    count = 0
    for j in data_clean[data_clean.columns[i]]:
        if j < min_t or j > max_t:
            count +=1
    percentage = count / data_clean.shape[0]
    outliers[data.columns[i]] = "%.3f" % percentage

print(outliers)
# {'ID': '0.004', 'LIMIT_BAL': '0.011', 'SEX': '0.000', 'EDUCATION': '0.005', 'MARRIAGE': '0.005',
# 'AGE': '0.005', 'PAY_0': '0.005', 'PAY_2': '0.006', 'PAY_3': '0.005', 'PAY_4': '0.004', 'PAY_5':
# '0.023', 'PAY_6': '0.022', 'BILL_AMT1': '0.022', 'BILL_AMT2': '0.023', 'BILL_AMT3': '0.022',
# 'BILL_AMT4': '0.022', 'BILL_AMT5': '0.013', 'BILL_AMT6': '0.010', 'PAY_AMT1':
# '0.012', 'PAY_AMT2': '0.013', 'PAY_AMT3': '0.014', 'PAY_AMT4': '0.015', 'PAY_AMT5': '0.000'}

# This results in a dictionary containing each feature name as a key, with the value
# representing the proportion of outliers for that feature. From these results, it is
# possible to observe that the features containing more outliers are BILL_AMT1
# and BILL_AMT4, each with a participation of 2.3% out of the total instances.
# This means that there are no further actions required given that the participation
# of outliers for each feature is too low, so they are unlikely to have an effect on
# the final model.

# Check for class imbalance: Class imbalance occurs when the class labels in the
# target feature are not equally represented; for instance, a dataset containing
# 90% of customers who did not default on the next payment, against 10% of
# customers who did, is considered to be imbalanced.
# There are several ways to handle class imbalance, some of which are
# explained here

# Collecting more data: Although this is not always an available route, it may help
# balance out the classes or allow for the removal of the over-represented class
# without reducing the dataset severely.
# Changing performance metrics: Some metrics, such as accuracy, are not
# good for measuring the performance of imbalanced datasets. In turn, it is
# recommended to measure performance using metrics such as precision or recall
# for classification problems.
# Resampling the dataset: This entails modifying the dataset to balance out the
# classes. This can be done in two different ways: adding copies of instances within
# under-represented class (called oversampling) or deleting instances of the overrepresented class (called undersampling).
# Class imbalance can be detected by simply counting the occurrences of each of
# the classes in the target feature, as shown here:
target = data_clean["default payment next month"]
yes = target[target == 1].count()  # 第一种
no = (target == 0).sum()  # 第二种


print("yes %:" + str(yes/len(target)*100) + " - no %:" + str(no/len(target) * 100))

print(target.value_counts()/len(target))  # 第三种

# From the preceding code, it is possible to conclude that the number of
# customers who defaulted on payments represents 22.12% of the dataset.
# These results can also be displayed in a plot using the following lines of code:
# target.value_counts().plot(kind='bar')

import matplotlib.pyplot as plt

fig, ax =plt.subplots(figsize=(10, 5))
plt.bar('yes', yes)
plt.bar('no', no)
ax.set_yticks([yes, no])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# In order to fix this problem, and in view of the fact that there is no more data to
# be added and that the performance metric is, in fact, accuracy, it is necessary to
# perform data resampling

data_yes = data_clean[data_clean["default payment next month"] == 1]
data_no = data_clean[data_clean["default payment next month"] == 0]

over_sampling = data_yes.sample(no, replace=True, random_state=0)
data_resampled = pd.concat([data_no, over_sampling], axis=0)

# 这行代码执行了过采样操作，具体解释如下：
#
# - `underrepresented_data` 是原始数据集中属于欠表示类别的数据。
# - `len(underrepresented_data)` 返回欠表示类别数据的数量。
# - `resampling_factor` 是重采样倍数，表示我们希望过采样后的数据量是原始欠表示类别数据量的多少倍。
# - `replace=True` 表示在进行采样时可以进行替换，即允许同一行数据被重复采样多次。
# - `n=len(underrepresented_data) * resampling_factor` 是采样的总数量，即希望获得的过采样后的数据总量。
#
# 因此，这一行代码的作用是从欠表示类别的数据中随机抽样，生成一个新的数据集，使其数量达到原始数据集中欠表示类别数据数量的 `resampling_factor` 倍。这就是实现过采样的关键步骤，目的是平衡数据集中各个类别的样本量。

target_2 = data_resampled["default payment next month"]
yes_2 = target_2[target_2 == 1].count()
no_2 = target_2[target_2 == 0].count()

print("yes %:" + str(yes_2/len(target_2)*100) + "- no %:" + str(no_2/len(target_2)*100))

print(data_resampled.shape)
# First, we separate the data for each class label into independent DataFrames.
# Next, we use the Pandas' sample() function to construct a new DataFrame
# that contains as many duplicated instances of the under-represented class as
# the over-represented class's DataFrame has.

# Finally, the concat() function is used to concatenate the DataFrame of the
# over-represented class and the newly created DataFrame of the same size, in
# order to create the final dataset to be used in subsequent steps.
# Using the newly created dataset, it is possible to, once again, calculate the
# participation of each class label in the target feature over the entire dataset,
# which should now reflect an equally represented dataset with both class labels
# having the same participation. The final shape of the dataset, at this point,
# should be equal to (46728, 23).

# Split features from target: We split the dataset into a features matrix and a
# target matrix to avoid rescaling the target values:
data_resampled = data_resampled.reset_index(drop=True)
X = data_resampled.drop(columns=['default payment next month'])
y = data_resampled['default payment next month']


# Rescaling the data: Finally, we rescale the values of the features matrix in order
# to avoid introducing bias to the model:
X = (X - X.min()) / (X.max() - X.min())
print(X.head())
# With the purpose of facilitating the use of the prepared dataset for the upcoming
# activities, both the features (X) and target (y) matrices will be concatenated into one
# Pandas DataFrame, which will be saved into a CSV file using the following code:

final_data = pd.concat([X, y], axis=1)
print(final_data.head())

final_data.to_csv("dcc_prepared.csv", index=False)
# After performing the preceding steps, the DCCC dataset has been transformed and is
# ready (in a new CSV file) to be used for training the model, which will be explained in
# the following section.

