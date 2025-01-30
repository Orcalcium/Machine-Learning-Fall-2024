from libsvm.svmutil import *

# Filter only classes 2 and 6 for binary classification
def filter_classes(y, x, classes=[3, 7]):
    indices = [i for i in range(len(y)) if y[i] in classes]
    y_filtered = [1 if y[i] == classes[0] else -1 for i in indices]  # Map classes to +1 and -1
    x_filtered = [x[i] for i in indices]
    return y_filtered, x_filtered

# Load data
y, x = svm_read_problem('./mnist.scale')
y, x = filter_classes(y, x)

Q_values = [2, 3, 4]
C_values = [0.1, 1, 10]
results = []
for Q in Q_values:
    for C in C_values:
        print(f'C={C} Q={Q}:')
        # Train SVM
        param = f'-t 1 -d {Q} -r 1 -c {C} -g 1 -q'
        model = svm_train(y, x, param)
        # Get number of support vectors
        support_vectors = model.get_SV()
        results.append((C, Q, len(support_vectors)))


# Display the results
print("C\tQ\tnumber of support vectors")
for C, Q, number_of_support_vectors in results:
    print(f"{C}\t{Q}\t{number_of_support_vectors}")


"""
C       Q       number of support vectors
-----------------------------------------
0.1     2       2581
1       2       1073
10      2       586
0.1     3       2141
1       3       916
10      3       535
0.1     4       1867
1       4       823
10      4       502
-----------------------------------------
"""

"""
C       Q       number of support vectors
-----------------------------------------
0.1     2       505
1       2       505
10      2       505
0.1     3       547
1       3       547
10      3       547
0.1     4       575
1       4       575
10      4       575
-----------------------------------------
"""