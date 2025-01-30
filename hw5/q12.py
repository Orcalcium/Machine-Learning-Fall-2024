import numpy as np
import matplotlib.pyplot as plt
from liblinear.liblinearutil import *
import random
from math import *
import time

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
start_time = time.time()

for i in range(1126):  # Run with different random seeds
    random.seed(i)  # Set a new seed
    print(f'this is the {i}th round')
    best_log_lambda = -float('inf')
    min_E_cv = float('inf')

    for log_lambda in lambda_values:
        lambda_value = 10 ** log_lambda
        C = 1 / lambda_value  # LIBLINEAR uses C, which is the inverse of lambda
        
        # Train the model and calculate E_cv
        model_cv = train(y_train, x_train, f'-s 6 -c {C} -B 1 -v 3 -q')
        # E_val is the 0/1 error (100 - accuracy)
        E_cv = 100 - model_cv
        if E_cv <= min_E_cv:
            best_log_lambda = log_lambda
            min_E_cv = E_cv

    print(f'Best log10(lambda): {best_log_lambda}')

    best_lambda_value = 10 ** best_log_lambda
    best_C = 1 / best_lambda_value  # LIBLINEAR uses C, which is the inverse of lambda
    model = train(y_train, x_train, f'-s 6 -c {best_C} -B 1 -q')
    p_label, p_acc, p_val = predict(y_test, x_test, model)
    E_out = 100 - p_acc[0]
    E_out_list.append(E_out)

end_time = time.time()
print(f'processing time is {end_time - start_time}')

plt.hist(E_out_list, bins=30, edgecolor='black')
plt.title(f'Histogram of E_out(g)')
plt.xlabel('E_out (%)')
plt.ylabel('Frequency')
plt.show()
