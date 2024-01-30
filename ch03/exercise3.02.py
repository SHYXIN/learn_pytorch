# Performance error analysis
# Using the accuracy metric we calculated in the previous activity, in this activity, we will
# perform error analysis, which will help us determine the actions to be performed over
# the model in the upcoming activity. Follow these steps to complete this exercise:
# 1. Assuming a Bayes error of 0.15, perform error analysis and diagnose
# the model:
Bayes_error = 0.15   # BE
training_set_error = 1 - 0.716  # 0.284   TSE
validation_set_error = 1 - 0.716 # 0.29   VSE
# The values that are being used as the accuracy of both sets (0.716 and 0.71)
# are the ones that were obtained during the last iteration of Activity 3.01, Building
# an ANN:
high_bias = training_set_error - Bayes_error # TSE - BE  = 0.134
high_variance = validation_set_error - training_set_error  # VSE - BE = 0.014
# According to this, the model is suffering from high bias, meaning that the model
# is underfitted.

# 2. Determine the actions necessary to improve the accuracy of the model.
# To improve the model's performance, the two courses of action that can be
# followed are incrementing the number of epochs and increasing the number of
# hidden layers and/or the number of units (neurons in each layer).
# In accordance with this, a set of tests can be performed in order to arrive at the
# best result