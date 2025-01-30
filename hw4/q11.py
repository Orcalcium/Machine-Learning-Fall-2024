import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

def polynomial_transform(X, Q):
    n_samples, n_features = X.shape
    transformed_features = [np.ones((n_samples, 1))]  # Bias term as a column vector

    for degree in range(1, Q + 1):
        degree_features = []
        for i in range(n_features):
            new_feature = (X[:, i] ** degree).reshape(-1, 1)  # Make it a column vector
            degree_features.append(new_feature)
        
        transformed_features.append(np.hstack(degree_features))  # Stack features for this degree

    # Concatenate all degree features horizontally
    return np.hstack(transformed_features)

# Load the dataset using load_svmlight_file
X_sparse, y = load_svmlight_file('cpusmall_scale.txt')

# Convert sparse matrix to dense format (if necessary)
X = X_sparse.toarray()

X_transformed = polynomial_transform(X, 3)

# Add x0 = 1 to each example (intercept term)
X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add x0 = 1

# Number of experiments
num_experiments = 1126
# Number of samples to select
N = 64

Ein_diff_list = []  # List to store Esqr_in(wlin) - Esqr_in(wpoly) in each run

# Helper function to compute squared error
def squared_error(X, y, w):
    predictions = X.dot(w)
    return np.mean((predictions - y) ** 2)

# Start experiments
for _ in range(num_experiments):
    # Randomly sample N examples from the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=N, test_size=X.shape[0]- N)
    X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_transformed, y, train_size=N, test_size=X_transformed.shape[0]- N)

    # Compute wlin and w_poly using pseudoinverse
    w_lin = np.linalg.pinv(X_train).dot(y_train)
    w_poly = np.linalg.pinv(X_poly_train).dot(y_poly_train)
    
    # Compute Ein and Eout
    Ein_lin = squared_error(X_train, y_train, w_lin)
    Ein_poly = squared_error(X_poly_train, y_poly_train, w_poly)

    # Calculate the difference in Ein and store in Esqr_diff_list
    Ein_diff = Ein_lin - Ein_poly
    Ein_diff_list.append(Ein_diff)

# Calculate the average Ein gain
average_Ein_gain = np.mean(Ein_diff_list)
print("Average E_in_lin - E_in_poly (E_in gain):", average_Ein_gain)

# Plot a histogram of Esqr_diff_list
plt.figure(figsize=(10, 5))
plt.hist(Ein_diff_list, bins=20, edgecolor='black')
plt.title("Histogram of E_in_lin - E_in_poly")
plt.xlabel("E_in_lin - E_in_poly")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
