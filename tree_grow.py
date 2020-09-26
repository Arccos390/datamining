from best_split import best_split
from anytree import AnyNode, RenderTree
import numpy as np

# Settings
_nmin = 10

# Global nodes
nodes = {}
memory_x = {}


# @todo
# def tree_grow(x, y, nmin, minleaf, nfeat):
#     # @todo make a check with y
#     number_of_col = np.shape(dataset)[1]
#     for j in range(number_of_col - 1):
#         target_row, target_col, target_value = best_split(x)
#         # select the majority
#         number_of_class_a = np.count_nonzero(x[x[:, target_col] <= target_value, -1])
#         number_of_class_b = len(x[x[:, target_col] <= target_value, -1]) - number_of_class_a
#         if j == 0:
#             root = Node('X%d <= %d' % (target_col, target_value))
#             child = Node((number_of_class_a, number_of_class_b), parent=root)
#             temp_x = x[x[:, target_col] <= target_value, :]
#             tree = tree_grow(temp_x, y, nmin, minleaf, nfeat)
#         elif j == 1:
#             child = Node('X%d <= %d' % (target_col, target_value), parent=root)
#             exec(f'child_{j - 1}_right = child')
#             child = Node((number_of_class_a, number_of_class_b), parent=child)
#             exec(f'child_{j}_left = child')
#         else:
#             child = Node('X%d <= %d' % (target_col, target_value), parent=child)
#             exec(f'child_{j - 1}_right = child')
#             child = Node((number_of_class_a, number_of_class_b), parent=child)
#             exec(f'child_{j}_left = child')
#
#         # @todo maybe we might split in the same column
#         x[:, target_col] = 0
#         # print(x)
#
#     return root

# Print a decision tree
def print_tree(t):
    print(RenderTree(t))


# if rule = none then leaf node
def tree_grow(x, y, nmin = _nmin, minleaf=None, nfeat=None):
    # if nmin not in memory_x:
    #     memory_x[nmin] = x
    # else:
    #     x = memory_x[nmin]
    # print(x)
    # print(memory_x)
    target_row, target_col, target_value = best_split(x)
    # select the majority
    number_of_class_a_split_1 = np.count_nonzero(x[x[:, target_col] <= target_value, -1])
    number_of_class_b_split_1 = len(x[x[:, target_col] <= target_value, -1]) - number_of_class_a_split_1

    number_of_class_a_split_2 = np.count_nonzero(x[x[:, target_col] > target_value, -1])
    number_of_class_b_split_2 = len(x[x[:, target_col] > target_value, -1]) - number_of_class_a_split_2

    if _nmin == nmin:
        root = AnyNode(id='root', rule=('X%d <= %.5f' % (target_col, target_value)))
        nodes[root.id] = root
        child_left = AnyNode(id='c1_1', parent=root, rule=None,
                             value=(number_of_class_a_split_1, number_of_class_b_split_1))
        child_right = AnyNode(id='c1_2', parent=root, rule=None,
                              value=(number_of_class_a_split_2, number_of_class_b_split_2))
    else:
        height = _nmin - nmin + 1
        width = 1
        # @todo make log steps to reduce time complexity
        for key in reversed(nodes.keys()):
            if key == 'root':
                continue
            temp_height = int(key.split('_')[0].replace('c', ''))
            if temp_height == height:
                width = int(key.split('_')[1])
        a = 0
        if (nodes['c%d_%d' % (height - 1, (width + 1) / 2)].value[0] == 0 or
                nodes['c%d_%d' % (height - 1, (width + 1) / 2)].value[1] == 0):
            a = 1
        child_left = AnyNode(id='c%d_%d' % (height, width), parent=nodes['c%d_%d' % (height - 1, (width + 1) / 2 + a)],
                             rule=None,
                             value=(number_of_class_a_split_1, number_of_class_b_split_1))
        child_right = AnyNode(id='c%d_%d' % (height, width + 1),
                              parent=nodes['c%d_%d' % (height - 1, (width + 1) / 2 + a)],
                              rule=None,
                              value=(number_of_class_a_split_2, number_of_class_b_split_2))
        nodes['c%d_%d' % (height - 1, (width + 1) / 2 + a)].rule = ('X%d <= %.5f' % (target_col, target_value))
    nodes[child_left.id] = child_left
    nodes[child_right.id] = child_right

    print_tree(nodes['root'])
    # If yes we can split again (left split)
    if number_of_class_a_split_1 != 0 and number_of_class_b_split_1 != 0:
        nmin -= 1
        if nmin != 0:
            x = x[x[:, target_col] <= target_value, :]
            tree_grow(x, y, nmin, minleaf, nfeat)

    # If yes we can split again (right split)
    if number_of_class_a_split_2 != 0 and number_of_class_b_split_2 != 0:
        nmin -= 1
        if nmin != 0:
            x = x[x[:, target_col] > target_value, :]
            tree_grow(x, y, nmin, minleaf, nfeat)

    nmin += 2
    print('s')
    print(nmin)
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
    [36, 1, 0, 52, 1, 1],
    [23, 0, 1, 40, 0, 1],
    [50, 1, 1, 28, 0, 1]
]


dataset = np.array(credit_data)
tree = tree_grow(dataset, [0, 1])
print(nodes)
print_tree(tree)
