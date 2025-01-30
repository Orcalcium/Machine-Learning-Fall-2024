import numpy as np
import matplotlib.pyplot as plt

# Function to parse the sparse-format dataset
def parse_sparse_data(filename):
    data, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            elements = line.strip().split()
            labels.append(int(elements[0]))
            features = {int(pair.split(':')[0]): float(pair.split(':')[1]) for pair in elements[1:]}
            data.append(features)
    
    # Convert to dense matrix
    num_features = 500
    dense_data = np.zeros((len(data), num_features))
    for i, features in enumerate(data):
        for index, value in features.items():
            dense_data[i, index - 1] = value
    return dense_data, np.array(labels)

# Function to find the best decision stump per dimension
def find_best_stump(X, y, weights):
    n_samples, n_features = X.shape
    best_stump = None
    best_error = float('inf')
    best_predictions = None

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            for direction in [-1, 1]:
                predictions = direction * np.sign(X[:, feature_index] - threshold)
                error = np.sum(weights[y != predictions])

                if error < best_error:
                    best_error = error
                    best_stump = {'feature': feature_index, 'threshold': threshold, 'direction': direction}
                    best_predictions = predictions

    return best_stump, best_error, best_predictions

# AdaBoost with modifications for no normalization
def adaboost(X_train, y_train, X_test, y_test, T):
    n_samples, n_features = X_train.shape

    # Initialize weights
    weights = np.ones(n_samples)/ n_samples
    alpha = []
    classifiers = []
    
    # Metrics to store
    ein_g_t = []  # E_in for g_t
    epsilons = []  # Normalized epsilon_t
    U_t = []  # Total weight sum
    E_in_values = []  # E_in of G_t
    E_out_values = []  # E_out of G_t

    # Train
    for t in range(T):
        print(f'this is the {t}-th iteration')
        stump, error, predictions = find_best_stump(X_train, y_train, weights)

        # Calculate alpha
        epsilon_t = max(error / np.sum(weights), 1e-10)  # Normalize for calculation only
        alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
        alpha.append(alpha_t)
        classifiers.append(stump)
        
        # Update weights (no normalization)
        weights = weights * np.exp(-alpha_t * y_train * predictions)
        U_t.append(np.sum(weights))  # Sum of weights

        # Record metrics
        ein_g_t.append(np.mean(predictions != y_train))
        epsilons.append(epsilon_t)

        # Calculate G_t and its errors
        G_train = np.sign(sum(alpha[j] * classifiers[j]['direction'] * 
                              np.sign(X_train[:, classifiers[j]['feature']] - classifiers[j]['threshold'])
                              for j in range(t + 1)))
        G_test = np.sign(sum(alpha[j] * classifiers[j]['direction'] * 
                             np.sign(X_test[:, classifiers[j]['feature']] - classifiers[j]['threshold'])
                             for j in range(t + 1)))
        E_in_values.append(np.mean(G_train != y_train))
        E_out_values.append(np.mean(G_test != y_test))

    return ein_g_t, epsilons, U_t, E_in_values, E_out_values

# Plot 1: Ein(g_t) and epsilon_t
def plot_ein_gt_eps(ein_g_t, epsilons):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(ein_g_t)), ein_g_t, label='Ein(g_t)', color='purple')
    plt.plot(range(len(epsilons)), epsilons, label='epsilon_t', color='orange')
    plt.xlabel('Iteration (t)')
    plt.ylabel('Error')
    plt.title('Ein(g_t) and epsilon_t over iterations')
    plt.legend()
    plt.grid()
    plt.show()

# Plot 2: E_in(G_t) and E_out(G_t)
def plot_ein_eout(E_in_values, E_out_values):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(E_in_values)), E_in_values, label='Ein(G_t)', color='blue')
    plt.plot(range(len(E_out_values)), E_out_values, label='Eout(G_t)', color='green')
    plt.xlabel('Iteration (t)')
    plt.ylabel('Error')
    plt.title('Ein(G_t) and Eout(G_t) over iterations')
    plt.legend()
    plt.grid()
    plt.show()

# Plot 3: U_t and Ein(G_t)
def plot_ut_ein(U_t, E_in_values):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(U_t)), U_t, label='U_t', color='red')
    plt.plot(range(len(E_in_values)), E_in_values, label='Ein(G_t)', color='blue')
    plt.xlabel('Iteration (t)')
    plt.ylabel('Value')
    plt.title('U_t and Ein(G_t) over iterations')
    plt.legend()
    plt.grid()
    plt.show()

# Load data
X_train, y_train = parse_sparse_data('madelon')
X_test, y_test = parse_sparse_data('madelon.t')

# Run AdaBoost and generate plots
T = 5
ein_g_t, epsilons, U_t, E_in_values, E_out_values = adaboost(X_train, y_train, X_test, y_test, T)

# Plot results
plot_ein_gt_eps(ein_g_t, epsilons)
plot_ein_eout(E_in_values, E_out_values)
plot_ut_ein(U_t, E_in_values)
