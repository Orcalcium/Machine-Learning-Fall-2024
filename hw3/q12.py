import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

# Load the dataset
data = load_svmlight_file("cpusmall_scale.txt")
X, y = data[0].toarray(), data[1]

# Use only the first two features (x1, x2) and add bias term x0 = 1
X_2_features = X[:, :2]  # Select only the first two features
X_2_features = np.hstack([np.ones((X_2_features.shape[0], 1)), X_2_features])  # Add x0 = 1

# List of N values
N_values = range(25, 2001, 25)
E_in_list = []
E_out_list = []

# Perform experiments
for N in N_values:
    E_in_exp = []
    E_out_exp = []
    
    for _ in range(16):  # Repeat each experiment 16 times
        # Randomly select N examples for training
        indices = np.random.choice(X.shape[0], N, replace=False)
        X_train, y_train = X_2_features[indices], y[indices]
        
        # Use the remaining examples for testing
        test_indices = np.setdiff1d(np.arange(X.shape[0]), indices)
        X_test, y_test = X_2_features[test_indices], y[test_indices]
        
        # Compute wlin using pseudoinverse
        w_lin = np.linalg.pinv(X_train).dot(y_train)
        
        # Calculate in-sample error (E_in)
        y_pred_train = X_train.dot(w_lin)
        E_in = np.mean((y_pred_train - y_train) ** 2)
        E_in_exp.append(E_in)
        
        # Calculate out-of-sample error (E_out)
        y_pred_test = X_test.dot(w_lin)
        E_out = np.mean((y_pred_test - y_test) ** 2)
        E_out_exp.append(E_out)
    
    # Average over 16 experiments
    E_in_list.append(np.mean(E_in_exp))
    E_out_list.append(np.mean(E_out_exp))

# Plotting the learning curves
plt.figure(figsize=(10, 6))
plt.plot(N_values, E_in_list, label='Average E_in', marker='o')
plt.plot(N_values, E_out_list, label='Average E_out', marker='x')
plt.xlabel('Number of training samples (N)')
plt.ylabel('Error')
plt.title('Learning Curves for E_in(N) and E_out(N)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(N_values[9:], E_in_list[9:], label='Average E_in', marker='o')
plt.plot(N_values[9:], E_out_list[9:], label='Average E_out', marker='x')
plt.xlabel('Number of training samples (N)')
plt.ylabel('Error')
plt.title('Learning Curves for E_in(N) and E_out(N)')
plt.legend()
plt.grid(True)
plt.show()