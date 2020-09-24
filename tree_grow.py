from best_split import best_split
from anytree import Node, RenderTree
import numpy as np


# @todo
def tree_grow(x, y, nmin, minleaf, nfeat):
    # @todo make a check with y
    number_of_col = np.shape(dataset)[1]
    for j in range(number_of_col - 1):
        target_row, target_col, target_value = best_split(x)
        # select the majority
        number_of_class_a = np.count_nonzero(x[x[:, target_col] <= target_value, -1])
        number_of_class_b = len(x[x[:, target_col] <= target_value, -1]) - number_of_class_a
        if j == 0:
            root = Node('X%d <= %d' % (target_col, target_value))
            child = Node((number_of_class_a, number_of_class_b), parent=root)
        elif j == 1:
            child = Node('X%d <= %d' % (target_col, target_value), parent=root)
            exec(f'child_{j-1}_right = child')
            child = Node((number_of_class_a, number_of_class_b), parent=child)
            exec(f'child_{j}_left = child')
        else:
            child = Node('X%d <= %d' % (target_col, target_value), parent=child)
            exec(f'child_{j - 1}_right = child')
            child = Node((number_of_class_a, number_of_class_b), parent=child)
            exec(f'child_{j}_left = child')

        x[:, target_col] = 0
        print(x)
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))
    return root


# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %d]' % ((depth * ' ', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))


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

dataset = np.array(credit_data)
tree_grow(dataset, [0, 1], 1, 1, 1)
