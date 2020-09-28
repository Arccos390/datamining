from best_split import best_split
from anytree import AnyNode, RenderTree
import numpy as np
from numpy import genfromtxt

# Global nodes
nodes = {}


# Print a decision tree
def print_tree(t):
    print(RenderTree(t))


# @todo make y separate from x and check if it 0 or 1
# @todo change the height and width parameter
# @todo fix the rule comparison. All the rules are <=. The values from the split are correct.
# @todo add image command to show the tree
# @todo nfeet
def tree_grow(x, y, nmin=10, minleaf=10, nfeat=None, height_parent=None, width_parent=None):
    target_row, target_col, target_value = best_split(x)
    number_of_class_a_split_1 = np.count_nonzero(x[x[:, target_col] <= target_value, -1])
    number_of_class_b_split_1 = len(x[x[:, target_col] <= target_value, -1]) - number_of_class_a_split_1

    number_of_class_a_split_2 = np.count_nonzero(x[x[:, target_col] > target_value, -1])
    number_of_class_b_split_2 = len(x[x[:, target_col] > target_value, -1]) - number_of_class_a_split_2

    if minleaf > number_of_class_a_split_1 + number_of_class_b_split_1:
        return
    if minleaf > number_of_class_a_split_2 + number_of_class_b_split_2:
        return

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

    # If yes we can split again (left split)
    if number_of_class_a_split_1 != 0 and number_of_class_b_split_1 != 0 and \
            nmin < number_of_class_a_split_1 + number_of_class_b_split_1:
        x_left = x[x[:, target_col] <= target_value, :]
        tree_grow(x_left, y, nmin, minleaf, nfeat, height_child, width_child)

    # If yes we can split again (right split)
    if number_of_class_a_split_2 != 0 and number_of_class_b_split_2 != 0 and \
            nmin < number_of_class_a_split_2 + number_of_class_b_split_2:
        x_right = x[x[:, target_col] > target_value, :]
        tree_grow(x_right, y, nmin, minleaf, nfeat, height_child, width_child + 1)

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

credit_data = [
    [22, 0, 0, 28, 1, 0],
    [46, 0, 1, 32, 0, 0],
    [24, 1, 1, 24, 1, 0],
    [25, 0, 0, 27, 1, 0],
    [29, 1, 1, 32, 0, 0],
    [45, 1, 1, 30, 0, 1],
    [63, 1, 1, 58, 1, 1],
    [63, 1, 1, 58, 1, 1],
    [63, 1, 1, 58, 1, 1],
    [63, 1, 1, 58, 1, 1],
    [63, 1, 1, 58, 0, 0],
    [36, 1, 0, 52, 1, 1],
    [23, 0, 1, 40, 0, 1],
    [50, 1, 1, 28, 0, 1]
]

# dataset = np.array(credit_data)

pima = genfromtxt('pima_numbers.csv', delimiter=',')

tree = tree_grow(pima, [0, 1])

print_tree(tree)
