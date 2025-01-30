import numpy as np
import matplotlib.pyplot as plt
from liblinear.liblinearutil import *
import random
from math import *
from sklearn.model_selection import train_test_split

# Filter only classes 2 and 6 for binary classification
def filter_classes(y, x, classes=[2, 6]):
    indices = [i for i in range(len(y)) if y[i] in classes]
    y_filtered = [1 if y[i] == classes[0] else -1 for i in indices]  # Map classes to +1 and -1
    x_filtered = [x[i] for i in indices]
    return y_filtered, x_filtered

# Read the training and testing data
y_train, x_train = svm_read_problem('mnist.scale')
y_test, x_test = svm_read_problem('mnist.scale.t')

y_train, x_train = filter_classes(y_train, x_train)
y_test, x_test = filter_classes(y_test, x_test)

# Range of lambda values to test (log10 scale)
lambda_values = [-2, -1, 0, 1, 2, 3]

E_out_list = []

for i in range(1):  # Run with different random seeds
    random.seed(i)  # Set a new seed
    print(f'this is the {i}th round')
    x_sub_train, x_validation, y_sub_train, y_validation = train_test_split(x_train, y_train, train_size=8000, test_size=len(x_train)-8000)
    best_log_lambda = -float('inf')
    min_E_val = float('inf')
    best_model = None
    for log_lambda in lambda_values:
        lambda_value = 10 ** log_lambda
        C = 1 / lambda_value  # LIBLINEAR uses C, which is the inverse of lambda
        
        # Train the model and calculate E_in
        model = train(y_sub_train, x_sub_train, f'-s 6 -c {C} -B 1 -q')
        p_label, p_acc, p_val = predict(y_validation, x_validation, model)
        # E_val is the 0/1 error (100 - accuracy)
        E_val = 100 - p_acc[0]
        if E_val <= min_E_val:
            best_log_lambda = log_lambda
            min_E_val = E_val
            best_model = model

    print(f'Best log10(lambda): {best_log_lambda}')
    p_label, p_acc, p_val = predict(y_test, x_test, best_model)
    E_out = 100 - p_acc[0]
    E_out_list.append(E_out)

plt.hist(E_out_list, bins=30, edgecolor='black')
plt.title(f'Histogram of E_out(g)')
plt.xlabel('E_out (%)')
plt.ylabel('Frequency')
plt.show()
