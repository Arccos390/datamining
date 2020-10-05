import numpy as np
import random
import pandas as pd


def best_split_n(x, y, nmin, minleaf, nfeat):  # x,y inputs + plus overfitting parameters
    parent_impurity = gini_index(y)
    number_of_col = x.shape[1]
    print(number_of_col)
    target = None
    col_target = None
    best_quality = 0
    # if parent has lower instances than nmin this node becomes a leaf -- return 2 Nones
    if x.shape[0] < nmin:
        return (target, col_target)
    # check nfeat parameter. Either all features will be examined or a random sample of them with length of nfeat
    if nfeat == number_of_col:
        feat = list(range(number_of_col))
    else:
        feat = random.sample(range(0, number_of_col), nfeat)
    for j in feat:  # examine each feature
        xcolumn = x[:, j]
        x_sorted = np.sort(np.unique(xcolumn))
        x_splitpoints = (x_sorted[0:x_sorted.shape[0] - 1] + x_sorted[1:x_sorted.shape[0]]) / 2
        qualities = np.array([])
        for point in x_splitpoints:  # examine all possible splitpoints
            child1 = y[xcolumn > point]
            child2 = y[xcolumn <= point]
            # check if the split is possible according to minleaf constrains
            if child1.shape[0] < minleaf or child2.shape[0] < minleaf:
                continue
            ratio = child1.shape[0] / y.shape[0]
            qualities = np.append(qualities,
                                  parent_impurity - ratio * gini_index(child1) - (1 - ratio) * gini_index(child2))
        candidate = np.max(qualities)
        ind = np.argmax(qualities)
        if candidate > best_quality:
            best_quality = candidate
            target = ind
            col_target = j
    return col_target, target


# Calculate the Gini index for a split dataset
def gini_index(labels):
    numerator = np.sum(labels)
    div = labels.shape[0]
    return (numerator / div) * (1 - numerator / div)


# test best_split
credit_data = [
    [45, 1, 1, 30, 0, 1],
    [63, 1, 1, 58, 1, 1],
    [63, 1, 0, 58, 1, 0],
    [36, 1, 0, 52, 1, 1],
    [50, 1, 1, 28, 0, 1]
]

dataset = [[2.771244718, 1.784783929, 0],
           [1.728571309, 1.169761413, 0],
           [3.678319846, 2.81281357, 0],
           [3.961043357, 2.61995032, 0],
           [2.999208922, 2.209014212, 0],
           [7.497545867, 3.162953546, 1],
           [9.00220326, 3.339047188, 1],
           [7.444542326, 0.476683375, 1],
           [10.12493903, 3.234550982, 1],
           [6.642287351, 3.319983761, 1]]

data = pd.DataFrame(credit_data)
x = data.iloc[:, 0:5]
y = data.iloc[:, 5]
print(np.array(x))
print(best_split_n(np.array(x), np.array(y), 10, 3, 4))
