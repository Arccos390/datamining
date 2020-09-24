from gini_index import gini_index
import numpy as np


def best_split(dataset):
    number_of_col = np.shape(dataset)[1]
    number_of_row = np.shape(dataset)[0]
    _min = 1
    target = None
    col_target = None
    row_target = None
    for j in range(number_of_col - 1):
        # todo if it is a class column skip the inner loop
        for i in range(number_of_row):
            temp = np.c_[dataset[:, j], dataset[:, -1]]
            target_for_split = dataset[i, j]
            gini = gini_index(temp, target_for_split)
            if gini is None:
                continue
            if gini < _min:
                _min = gini
                target = dataset[i, j]
                row_target = i
                col_target = j
    print(_min)
    print(target)
    print(row_target)
    print(col_target)
    return row_target, col_target, target


# test best_split
credit_data = [
    [22, 0, 0, 28, 1, 0],
    [46, 0, 1, 32, 0, 0],
    [24, 1, 1, 24, 1, 0],
    [25, 0, 0, 27, 1, 0],
    [29, 1, 1, 32, 0, 0],
    [45, 1, 1, 30, 0, 1],
    [63, 1, 1, 58, 1, 1],
    [36, 1, 0, 52, 1, 1],
    [23, 0, 1, 40, 0, 1],
    [50, 1, 1, 28, 0, 1]
]


print(best_split(np.array(credit_data)))
