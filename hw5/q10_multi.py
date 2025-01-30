import numpy as np
import matplotlib.pyplot as plt
from liblinear.liblinearutil import *
import random
from math import *
import multiprocessing
import time

# Filter only classes 2 and 6 for binary classification
def filter_classes(y, x, classes=[2, 6]):
    indices = [i for i in range(len(y)) if y[i] in classes]
    y_filtered = [1 if y[i] == classes[0] else -1 for i in indices]  # Map classes to +1 and -1
    x_filtered = [x[i] for i in indices]
    return y_filtered, x_filtered

# Experiment function to be run in parallel
def run_experiment(i):
    print(f'this is the {i}th round')
    
    best_log_lambda = -float('inf')
    min_E_in = float('inf')
    best_W = []
    best_model = None
    
    # Loop over lambda values
    for log_lambda in lambda_values:
        lambda_value = 10 ** log_lambda
        C = 1 / lambda_value  # LIBLINEAR uses C, which is the inverse of lambda
        
        # Train the model and calculate E_in
        model = train(y_train, x_train, f'-s 6 -c {C} -B 1 -q')
        p_label, p_acc, p_val = predict(y_train, x_train, model)
        [W, b] = model.get_decfun()
        
        # E_in is the 0/1 error (100 - accuracy)
        E_in = 100 - p_acc[0]
        if E_in <= min_E_in:
            best_log_lambda = log_lambda
            min_E_in = E_in
            best_W = W
            best_model = model
    
    print(f'Best log10(lambda): {best_log_lambda}')
    # Evaluate on test set
    p_label, p_acc, p_val = predict(y_test, x_test, best_model)
    E_out = 100 - p_acc[0]
    
    # Return the results
    return E_out, np.count_nonzero(best_W)

# Read the training and testing data
y_train, x_train = svm_read_problem('mnist.scale')
y_test, x_test = svm_read_problem('mnist.scale.t')

y_train, x_train = filter_classes(y_train, x_train)
y_test, x_test = filter_classes(y_test, x_test)

# Range of lambda values to test (log10 scale)
lambda_values = [-2, -1, 0, 1, 2, 3]

# Number of experiments to run
num_experiments = 2

if __name__ == '__main__':

    start_time = time.time()
    # Use multiprocessing to run experiments in parallel
    with multiprocessing.Pool() as pool:
        results = pool.map(run_experiment, range(num_experiments))

    # Extract the results
    E_out_list, non_zero_components_list = zip(*results)

    end_time = time.time()
    print(f'processing time is {end_time - start_time}')

    # Plot histogram of E_out
    plt.hist(E_out_list, bins=30, edgecolor='black')
    plt.title(f'Histogram of E_out(g)')
    plt.xlabel('E_out (%)')
    plt.ylabel('Frequency')
    plt.show()

    # Plot histogram of non-zero components
    plt.hist(non_zero_components_list, bins=30, edgecolor='black')
    plt.title(f'Histogram of E_out(g)')
    plt.xlabel('Number of Non-zero Components')
    plt.ylabel('Frequency')
    plt.show()
