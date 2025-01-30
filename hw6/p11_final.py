from libsvm.svmutil import *
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import time

# Filter only classes 3 and 7 for binary classification
def filter_classes(y, x, classes=[3, 7]):
    indices = [i for i in range(len(y)) if y[i] in classes]
    y_filtered = [1 if y[i] == classes[0] else -1 for i in indices]  # Map classes to +1 and -1
    x_filtered = [x[i] for i in indices]
    return y_filtered, x_filtered

def sparse_to_dense(sparse_dict, size):
    dense_array = [0] * (size + 1)
    for index, value in sparse_dict.items():
        dense_array[index] = value
    return dense_array

# Load data
y, x = svm_read_problem('./mnist.scale')
y, x = filter_classes(y, x)

# Initialize a variable to track the largest key
max_key = -1

# Iterate over each example in x (which is a list of dictionaries)
for xi in x:
    max_key = max(max_key, *xi.keys())  # Get the largest key for each example

# Parameter combinations
C_values = [0.1, 1, 10]
gamma_values = [0.1, 1, 10]
results = []

start = time.time()
for C in C_values:
    for gamma in gamma_values:
        print(f'C={C} gamma={gamma}:')
        param = f'-t 2 -c {C} -g {gamma} -q'  # Gaussian kernel (-t 2)
        model = svm_train(y, x, param)
        print("Model trained")

        # Extract support vectors indices and alphas
        sv = model.get_SV()
        sv_coef = model.get_sv_coef()

        # 預先將所有支持向量轉為密集向量
        sv_dense = [np.asarray(sparse_to_dense(s, max_key)) for s in sv]
        
        # 預先計算每個向量的平方和
        squared_norms = [np.linalg.norm(vec)**2 for vec in sv_dense]

        # Compute ||w||^2
        w_norm_sq = 0
        K = np.zeros((len(sv), len(sv)))

        # 計算核矩陣並累積 w 的平方範數
        for i, xi in enumerate(sv_dense):
            for j, xj in enumerate(sv_dense):
                # 利用預先計算的平方和快速計算距離
                dist_sq = squared_norms[i] + squared_norms[j] - 2 * np.dot(xi, xj)
                K[i, j] = np.exp(-gamma * dist_sq)
                w_norm_sq += sv_coef[i][0] * sv_coef[j][0] * K[i, j]

        margin = 1 / np.sqrt(w_norm_sq)
        results.append((C, gamma, margin))
        flag = time.time()
        print(f'current processing time = {flag - start}')

end = time.time()
print(f'total processing time = {end - start}')

# Display the results
print("C\tGamma\tMargin")
for C, gamma, margin in results:
    print(f"{C}\t{gamma}\t{margin}")

"""
C       Gamma   Margin
--------------------------------------
0.1     0.1     0.04528271745937934
0.1     1       0.09077972164855126
0.1     10      0.0907930846971181
1       0.1     0.02096789213401075
1       1       0.009077996936730314
1       10      0.00907933595784014
10      0.1     0.02064555588123403
10      1       0.008983588792855345
10      10      0.008981868111248254
--------------------------------------
"""