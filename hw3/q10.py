import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

# Load the dataset using load_svmlight_file
# Adjust the path to your actual file location
X_sparse, y = load_svmlight_file('cpusmall_scale.txt')

# Convert sparse matrix to dense format (if necessary)
X = X_sparse.toarray()

# Add x0 = 1 to each example (intercept term)
X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add x0 = 1

# Number of experiments
num_experiments = 1126
# Number of samples to select
N = 32

# Prepare arrays to store Ein and Eout
Ein_list = []
Eout_list = []

# Helper function to compute squared error
def squared_error(X, y, w):
    predictions = X.dot(w)
    return np.mean((predictions - y) ** 2)

# Start experiments
for _ in range(num_experiments):
    # Randomly sample N examples from the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=N, test_size=X.shape[0]-N)
    
    # Compute wlin using pseudoinverse
    w_lin = np.linalg.pinv(X_train).dot(y_train)

    # Compute Ein and Eout
    Ein = squared_error(X_train, y_train, w_lin)
    Eout = squared_error(X_test, y_test, w_lin)

    # Store the results
    Ein_list.append(Ein)
    Eout_list.append(Eout)

# Scatter plot of Ein vs Eout
plt.figure(figsize=(10,6))
plt.scatter(Ein_list, Eout_list, alpha=0.5)
plt.xlabel('Ein (In-sample error)')
plt.ylabel('Eout (Out-of-sample error)')
plt.title(f'Ein vs Eout Scatter Plot ({num_experiments} experiments)')
plt.show()

print("median Ein", np.median(Ein))
print("median Eout", np.median(Eout)) 