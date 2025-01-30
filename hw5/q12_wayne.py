import numpy as np
import matplotlib.pyplot as plt
from liblinear.liblinearutil import *

# Paths to training and testing data
train_file = "mnist.scale"  # Training file path
test_file = "mnist.scale.t"  # Testing file path

def load_data(file_path):
    y, x = svm_read_problem(file_path)
    
    # Filter for classes 2 and 6
    filtered_y = []
    filtered_x = []
    for label, features in zip(y, x):
        if label == 2 or label == 6:
            filtered_y.append(1 if label == 6 else -1)  # Convert labels: 6 -> +1, 2 -> -1
            filtered_x.append(features)
    
    return np.array(filtered_y), filtered_x

# Load data
y_train, x_train = load_data(train_file)
y_test, x_test = load_data(test_file)

# Hyperparameters: log10(λ) values
lambda_values = [-2, -1, 0, 1, 2, 3]

# Results storage
E_out_values = []

# Run 1126 experiments with different random seeds
for seed in range(1126):
    np.random.seed(seed)
    print("this is the ", seed, " round")
    # Shuffle training data
    indices = np.random.permutation(len(y_train))
    y_shuffled = y_train[indices]
    x_shuffled = [x_train[i] for i in indices]
    
    # Split data into 3 folds
    fold_size = len(y_train) // 3
    folds = [(y_shuffled[i * fold_size:(i + 1) * fold_size], 
              x_shuffled[i * fold_size:(i + 1) * fold_size]) for i in range(3)]
    
    # Find the best λ using 3-fold cross-validation
    best_lambda = None
    best_E_cv = float('inf')
    
    for log10_lambda in lambda_values:
        λ = 10 ** log10_lambda
        C = 1 / λ
        param_str = f'-s 6 -c {C} -B 1 -q'
        
        # Perform 3-fold cross-validation
        E_cv = 0
        for k in range(3):
            # Use two folds for training, one fold for validation
            y_train_cv = np.concatenate([folds[i][0] for i in range(3) if i != k])
            x_train_cv = sum([folds[i][1] for i in range(3) if i != k], [])
            y_val_cv = folds[k][0]
            x_val_cv = folds[k][1]
            
            # Train the model
            model = train(y_train_cv, x_train_cv, param_str)
            
            # Validate on the validation fold
            p_labels, _, _ = predict(y_val_cv, x_val_cv, model)
            error = np.mean(np.array(p_labels) != np.array(y_val_cv))
            E_cv += error
        
        # Average validation error across folds
        E_cv /= 3
        
        # Track the best λ
        if E_cv < best_E_cv or (E_cv == best_E_cv and log10_lambda > best_lambda):
            best_E_cv = E_cv
            best_lambda = log10_lambda
    
    # Re-train with the best λ on the entire training set
    λ_star = 10 ** best_lambda
    C_star = 1 / λ_star
    final_param_str = f'-s 6 -c {C_star} -B 1 -q'
    final_model = train(y_train, x_train, final_param_str)
    
    # Evaluate E_out on the test set
    _, p_acc, _ = predict(y_test, x_test, final_model)
    E_out = 100 - p_acc[0]  # Convert accuracy to error percentage
    E_out_values.append(E_out)

# Plot histogram of E_out
plt.figure(figsize=(8, 5))
plt.hist(E_out_values, bins=20, color='blue', alpha=0.7)
plt.title("Histogram of $E_{out}(g)$ over 1126 Experiments (3-Fold CV)")
plt.xlabel("$E_{out}(g)$ (%)")
plt.ylabel("Frequency")
plt.show()