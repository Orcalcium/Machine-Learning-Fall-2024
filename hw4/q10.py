import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

# Load the dataset
X_sparse, y = load_svmlight_file('cpusmall_scale.txt')
X = X_sparse.toarray()
X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add x0 = 1

# Parameters
num_experiments = 1126
N = 64 # Number of samples to select
num_iterations = 100000  # Total iterations for SGD
record_interval = 200  # Record every 200 iterations
learning_rate = 0.01

# Prepare arrays to store Ein and Eout for linear regression
Ein_list = []
Eout_list = []


# Prepare to collect averages over all experiments
Ein_sgd_average = np.zeros(num_iterations // record_interval)  # Average Ein
Eout_sgd_average = np.zeros(num_iterations // record_interval)  # Average Eout

# Helper function to compute squared error
def squared_error(X, y, w):
    predictions = X.dot(w)
    return np.mean((predictions - y) ** 2)

# Run experiments
for experiment in range(num_experiments):
    print(experiment)
    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=N, test_size=X.shape[0]- N)

    # Compute wlin using pseudoinverse for linear regression
    w_lin = np.linalg.pinv(X_train).dot(y_train)

    # Compute Ein and Eout for linear regression
    Ein = squared_error(X_train, y_train, w_lin)
    Eout = squared_error(X_test, y_test, w_lin)

    # Store results for linear regression
    Ein_list.append(Ein)
    Eout_list.append(Eout)

    # Initialize weights
    w_sgd = np.zeros(X_train.shape[1])

    # Lists to store Ein and Eout for this experiment
    Ein_sgd_temp = []
    Eout_sgd_temp = []

    for t in range(num_iterations):
        # Pick one example uniformly at random
        idx = np.random.randint(N)
        x_t = X_train[idx]
        y_t = y_train[idx]

        # Update weights using SGD
        w_sgd += learning_rate * 2 * (y_t - x_t.dot(w_sgd)) * x_t

        # Record Ein and Eout every 200 iterations
        if t > 0 and (t+1) % record_interval == 0:
            Ein_temp = squared_error(X_train, y_train, w_sgd)
            Eout_temp = squared_error(X_test, y_test, w_sgd)
            step_index = t // record_interval
            Ein_sgd_average[step_index] += Ein_temp
            Eout_sgd_average[step_index] += Eout_temp

# Calculate average Ein and Eout over the experiments for linear regression
average_Ein_lin = np.mean(Ein_list)
average_Eout_lin = np.mean(Eout_list)

# Final averaging over all experiments
Ein_sgd_average /= num_experiments
Eout_sgd_average /= num_experiments


# Prepare t values for plotting SGD results
t_values = np.arange(200, num_iterations + 1, record_interval)

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(t_values, Ein_sgd_average, label='Average Ein (SGD)', color='blue')
plt.plot(t_values, Eout_sgd_average, label='Average Eout (SGD)', color='orange')
plt.axhline(y=average_Ein_lin, color='green', linestyle='--', label='Average Ein (Linear)')
plt.axhline(y=average_Eout_lin, color='red', linestyle='--', label='Average Eout (Linear)')
plt.xlabel('Iterations (t)')
plt.ylabel('Error')
plt.title('SGD Regression: Average Ein and Eout Over Time')
plt.legend()
plt.grid()
plt.show()

print("Average Ein (Linear Regression):", average_Ein_lin)
print("Average Eout (Linear Regression):", average_Eout_lin)
print("Average Ein (SGD):", Ein_sgd_average)
print("Average Eout (SGD):", Eout_sgd_average)
