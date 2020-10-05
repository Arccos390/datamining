from anytree import AnyNode, RenderTree
import numpy as np
import pandas as pd
from numpy import genfromtxt
import random

# Global nodes
nodes = {}


# Print a decision tree
def print_tree(t):
    print(RenderTree(t))


def best_split(x, y):  # x,y inputs + plus overfitting parameters
    parent_impurity = gini_index(y)
    number_of_col = x.shape[1]
    row_target = None
    col_target = None
    best_quality = 0
    # # if parent has lower instances than nmin this node become
    # s a leaf -- return 2 Nones
    # if x.shape[0] < nmin:
    #     return (target, col_target)
    # check nfeat parameter. Either all features will be examined or a random sample of them with length of nfeat

    for j in range(number_of_col):  # examine each feature
        xcolumn = x[:, j]
        x_sorted = np.sort(np.unique(xcolumn))
        x_splitpoints = (x_sorted[0:x_sorted.shape[0] - 1] + x_sorted[1:x_sorted.shape[0]]) / 2
        qualities = np.array([])
        for point in x_splitpoints:  # examine all possible splitpoints
            child1 = y[xcolumn > point]
            child2 = y[xcolumn <= point]
            # # check if the split is possible according to minleaf constrains
            # if child1.shape[0] < minleaf or child2.shape[0] < minleaf:
            #     continue
            ratio = child1.shape[0] / y.shape[0]
            qualities = np.append(qualities,
                                  parent_impurity - ratio * gini_index(child1) - (1 - ratio) * gini_index(child2))
        if len(qualities) > 0:
            print(qualities)
            candidate = np.max(qualities)
            ind = np.argmax(qualities)
            if candidate > best_quality:
                best_quality = candidate
                row_target = ind
                col_target = j
    return row_target, col_target


# Calculate the Gini index for a split dataset
def gini_index(labels):
    numerator = np.sum(labels)
    div = labels.shape[0]
    return (numerator / div) * (1 - numerator / div)


# @todo change the height and width parameter
# @todo fix the rule comparison. All the rules are <=. The values from the split are correct.
# @todo add image command to show the tree
# @todo nfeet
# @todo minleaf
def tree_grow(x, y, nmin=5, minleaf=10, nfeat=None, height_parent=None, width_parent=None):
    # if nfeat == number_of_col:
    #     feat = list(range(number_of_col))
    # else:
    #     feat = random.sample(range(0, number_of_col), nfeat)
    # print(x)
    # print(y)
    target_row, target_col = best_split(x, y)
    target_value = x[target_row, target_col]
    print(target_value)
    number_of_class_a_split_1 = np.count_nonzero(y[x[:, target_col] <= target_value])
    number_of_class_b_split_1 = len(y[x[:, target_col] <= target_value]) - number_of_class_a_split_1

    number_of_class_a_split_2 = np.count_nonzero(y[x[:, target_col] > target_value])
    number_of_class_b_split_2 = len(y[x[:, target_col] > target_value]) - number_of_class_a_split_2

    # if minleaf > number_of_class_a_split_1 + number_of_class_b_split_1:
    #     return
    # if minleaf > number_of_class_a_split_2 + number_of_class_b_split_2:
    #     return

    if height_parent is None and width_parent is None:
        root = AnyNode(id='root', rule=('X%d <= %.5f' % (target_col, target_value)))
        nodes[root.id] = root
        child_left = AnyNode(id='c1_1', parent=root, rule=None,
                             value=(number_of_class_a_split_1, number_of_class_b_split_1))
        child_right = AnyNode(id='c1_2', parent=root, rule=None,
                              value=(number_of_class_a_split_2, number_of_class_b_split_2))
        height_child = 1
        width_child = 1
    else:
        height_child = height_parent + 1
        width_child = 1
        # @todo make log steps to reduce time complexity
        # Find if there is any node in the same height as the child's height. If yes continue the width of the last
        # child's node
        for key in reversed(nodes.keys()):
            if key == 'root':
                continue
            current_height = int(key.split('_')[0].replace('c', ''))
            if current_height == height_child:
                width_child = int(key.split('_')[1]) + 1
                break
        child_left = AnyNode(id='c%d_%d' % (height_child, width_child),
                             parent=nodes['c%d_%d' % (height_parent, width_parent)],
                             rule=None,
                             value=(number_of_class_a_split_1, number_of_class_b_split_1))
        child_right = AnyNode(id='c%d_%d' % (height_child, width_child + 1),
                              parent=nodes['c%d_%d' % (height_parent, width_parent)],
                              rule=None,
                              value=(number_of_class_a_split_2, number_of_class_b_split_2))
        nodes['c%d_%d' % (height_parent, width_parent)].rule = ('X%d <= %.5f' % (target_col, target_value))
    nodes[child_left.id] = child_left
    nodes[child_right.id] = child_right

    print_tree(nodes['root'])
    # If yes we can split again (left split)
    if number_of_class_a_split_1 != 0 and number_of_class_b_split_1 != 0 and \
            nmin < number_of_class_a_split_1 + number_of_class_b_split_1:
        x_left = x[x[:, target_col] <= target_value, :]
        y_left = y[x[:, target_col] <= target_value]
        tree_grow(x_left, y_left, nmin, minleaf, nfeat, height_child, width_child)

    # If yes we can split again (right split)
    if number_of_class_a_split_2 != 0 and number_of_class_b_split_2 != 0 and \
            nmin < number_of_class_a_split_2 + number_of_class_b_split_2:
        x_right = x[x[:, target_col] > target_value, :]
        y_right = y[x[:, target_col] > target_value]
        tree_grow(x_right, y_right, nmin, minleaf, nfeat, height_child, width_child + 1)

    return nodes['root']


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

# test best_split
credit_data = [
    [22, 0, 0, 28, 1, 0],
    [46, 0, 1, 32, 0, 0],
    [24, 1, 1, 24, 1, 0],
    [25, 0, 0, 27, 1, 0],
    [29, 1, 1, 32, 0, 0],
    [45, 1, 1, 30, 0, 1],
    [63, 1, 1, 58, 1, 1],
    [63, 1, 0, 58, 1, 0],
    [36, 1, 0, 52, 1, 1],
    [23, 0, 1, 40, 0, 1],
    [50, 1, 1, 28, 0, 1]
]


pima = genfromtxt('pima_numbers.csv', delimiter=',')
data = pd.DataFrame(credit_data)
x = data.iloc[:, 0:data.shape[1]-1]
y = data.iloc[:, data.shape[1]-1]
print(np.array(x))
tree = tree_grow(np.array(x), np.array(y), 2, 2, 1)

print_tree(tree)
