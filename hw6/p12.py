from libsvm.svmutil import *
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Filter only classes 3 and 7 for binary classification
def filter_classes(y, x, classes=[3, 7]):
    indices = [i for i in range(len(y)) if y[i] in classes]
    y_filtered = [1 if y[i] == classes[0] else -1 for i in indices]  # Map classes to +1 and -1
    x_filtered = [x[i] for i in indices]
    return y_filtered, x_filtered

# Load data
y, x = svm_read_problem('./mnist.scale')
y, x = filter_classes(y, x)

# Define gamma values
gamma_values = [0.01, 0.1, 1, 10, 100]
C = 1
selection_count = {gamma: 0 for gamma in gamma_values}

# Number of repetitions
repetitions = 128
val_size = 200

start = time.time()

for i in range(repetitions):
    print(f'this is {i}th round')
    # Split data into training and validation sets
    indices = list(range(len(y)))
    random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    y_train = [y[i] for i in train_indices]
    x_train = [x[i] for i in train_indices]
    y_val = [y[i] for i in val_indices]
    x_val = [x[i] for i in val_indices]
    
    # Track validation errors for each gamma
    val_errors = {}
    best_gamma = None
    min_error = float('inf')
    for gamma in gamma_values:
        param = f'-t 2 -c {C} -g {gamma} -q'
        model = svm_train(y_train, x_train, param)
        
        # Predict on validation set
        p_label, _, _ = svm_predict(y_val, x_val, model, '-q')
        
        # Calculate 0/1 error
        error = sum(1 for i in range(len(y_val)) if p_label[i] != y_val[i]) / len(y_val)
        if error < min_error:
            best_gamma = gamma
            min_error = error
    selection_count[best_gamma] += 1
    for gamma, count in selection_count.items():
        print(f"Gamma: {gamma}, Count: {count}")
    flag = time.time()
    print(f'current processing time = {flag - start}')

end = time.time()
print(f'processing time = {end - start}')
# Print selection frequencies
print("Selection frequencies for each gamma:")
for gamma, count in selection_count.items():
    print(f"Gamma: {gamma}, Count: {count}")
    
# Plot the bar chart with a logarithmic x-axis
plt.bar([np.log10(gamma) for gamma in selection_count.keys()], selection_count.values(), color='blue', width=0.2)
plt.xlabel('Gamma Values')
plt.ylabel('Selection Frequency')
plt.title('Selection Frequency of Gamma Values')
plt.xticks([np.log10(gamma) for gamma in gamma_values], labels=gamma_values)  # Show original gamma values as labels
plt.show()

