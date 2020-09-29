import random
from random import randint
from anytree import AnyNode, RenderTree
import numpy as np
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as conf

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
def tree_grow(x, y, nmin=8, minleaf=3, nfeat=None): #wrong parameters, more than we asked. It should work in a loop and not recursive
    if nfeat is None:
        nfeat = x.shape[1]
    parent = None
    #nodes_to_examine = lifo()
    nodes_to_examine = []
    node_counter = 2 # for iteration between nodes
    while True:
        target_col, target_value = best_split(x, y, nmin, minleaf, nfeat)  # parameters for overfitting should be passed on best split and examined there
        #count the number of instances per class for the child nodes
        #To access root
        if parent is None:
            root = AnyNode(id='root', rule=('X%d <= %.5f' % (target_col, target_value)), split_value=target_value, split_column = target_col)
            nodes[root.id] = root
            x_l = x[x[:, target_col] > target_value, :]
            x_r = x[x[:, target_col] <= target_value, :]
            y_l = []
            y_r = []
            helper = x[:, target_col]
            for i in range(len(helper)):
                if helper[i] > target_value:
                    y_l.append(y[i])
                else:
                    y_r.append(y[i])
            number_of_class_a_split_1 = sum(y_l)
            number_of_class_b_split_1 = len(y_l) - number_of_class_a_split_1
            number_of_class_a_split_2 =  sum(y_r)
            number_of_class_b_split_2 = len(y_r) - number_of_class_a_split_2
            child_left = AnyNode(id='c1', parent=root, rule=None, x=x_l, y=y_l,
                                 value=[number_of_class_a_split_1, number_of_class_b_split_1])
            child_right = AnyNode(id='c2', parent=root, rule=None, x= x_r, y = y_r,
                                  value=[number_of_class_a_split_2, number_of_class_b_split_2])
            #trial for pred
            root.children= [child_left,child_right]
            nodes[child_left.id] = child_left
            nodes[child_right.id] = child_right
            nodes_to_examine.append(child_left)
            nodes_to_examine.append(child_right)
            parent = 1
        #any other node
        else:
            #check if lifo empty, if it is break as there are no other nodes to examine
            if not nodes_to_examine:
                break
            #get node from lifo
            candidate = nodes_to_examine[0]
            nodes_to_examine = nodes_to_examine[1:]
            x_c = candidate.x #get the instances that belong to the node
            y_c = candidate.y
            #second check
            if x_c.shape[0] < nmin or x_c.shape[0] < 2*minleaf:
                continue
            target_col, target_value = best_split(x_c, y_c, nmin, minleaf,
                                                  nfeat)  #get the split for the node
            if target_value is None and target_col is None:
                continue
            else:
                node_counter = node_counter + 1
                x_l = x_c[x_c[:, target_col] > target_value, :]
                x_r = x_c[x_c[:, target_col] <= target_value, :]
                """
                if x_l.shape[0] < minleaf or x_r.shape[0] < minleaf:
                    break
                    """
                y_l = []
                y_r = []
                helper = x_c[:, target_col]
                for i in range(len(helper)):
                    if helper[i] > target_value:
                        y_l.append(y_c[i])
                    else:
                        y_r.append(y_c[i])
                number_of_class_a_split_1 = sum(y_l)
                number_of_class_b_split_1 = len(y_l) - number_of_class_a_split_1
                number_of_class_a_split_2 = sum(y_r)
                number_of_class_b_split_2 = len(y_r) - number_of_class_a_split_2
                child_left = AnyNode(id='c%d' % node_counter,
                                     parent=candidate, split_value = None, split_column = None,
                                     rule=None, x=x_l, y = y_l,
                                     value=[number_of_class_a_split_1, number_of_class_b_split_1])
                node_counter = node_counter + 1
                child_right = AnyNode(id='c%d' % node_counter,
                                      parent=candidate, split_value = None, split_column = None,
                                      rule=None, x=x_r, y = y_r,
                                      value=[number_of_class_a_split_2, number_of_class_b_split_2])
                candidate.rule = ('X%d <= %.5f' % (target_col, target_value))
                candidate.children = [child_left, child_right]
                candidate.split_column = target_col
                candidate.split_value = target_value
                nodes[child_left.id] = child_left
                nodes[child_right.id] = child_right
                nodes_to_examine.append(child_left)
                nodes_to_examine.append(child_right)
    return nodes['root']


def tree_pred(x, tree):
    predictions = []
    for xpred in x:
        pred = None
        p = tree
        while (True):
            if p.rule == None:
                if p.value[0] > p.value[1]:
                    predictions.append(1)
                    break
                else:
                    predictions.append(0)
                    break
            else:
                col = p.split_column
                val = p.split_value
                if xpred[col] > val:
                    new_p = p.children[0]
                else:
                    new_p = p.children[1]
                p = new_p
    return predictions


def tree_grow_b(x, y, m, nmin=8, minleaf=3, nfeat=None):
    trees = []
    dim = x.shape[0]
    for i in range(m):
        x_local = []
        y_local = []
        for j in range(dim):
            ind = randint(0,dim-1)
            x_local.append(x[ind,:])
            y_local.append(y[ind])
        tree = tree_grow(x,y,nmin,minleaf,nfeat)
        trees.append(tree)
    return trees

def tree_pred_b(x, trees):
    predictions_bag = []
    predictions_gathered = []
    m = len(trees)
    for i in range(m):
        predictions = tree_pred(x, trees[i])
        predictions_gathered.append(predictions)
    for j in range(x.shape[0]):
        summed = 0
        for k in range(m):
            summed = summed + predictions_gathered[k][j]
        if summed/m > 0.5 :
            predictions_bag.append(1)
        else:
            predictions_bag.append(0)
    return predictions_bag


def best_split(x, y, nmin, minleaf,  nfeat): #x,y inputs + plus overfitting parameters
    parent_impurity = gini_index(y)
    number_of_col = x.shape[1]
    target = None
    col_target = None
    best_quality = 0
    # if parent has lower instances than nmin this node becomes a leaf -- return 2 Nones
    if x.shape[0] < nmin:
        return target, col_target
    #check nfeat parameter. Either all features will be examined or a random sample of them with length of nfeat
    if nfeat == number_of_col:
        feat = list(range(number_of_col))
    else:
        feat = random.sample(range(0, number_of_col - 1), nfeat)
    for j in feat:# examine each feature
        xcolumn = x[:, j]
        x_sorted = np.sort(np.unique(xcolumn))
        x_splitpoints = []
        for i in range(x_sorted.shape[0]-1):
            x_splitpoints.append((x_sorted[i]+x_sorted[i+1])/2)
        qualities = np.array([])
        x_kept = []
        for point in x_splitpoints: # examine all possible splitpoints
            child1 = []
            child2 = []
            for i in range(len(xcolumn)):
                if xcolumn[i] > point:
                    child1.append(y[i])
                else:
                    child2.append(y[i])
            # check if the split is possible according to minleaf constrains
            if len(child1) < minleaf or len(child2) < minleaf:
                continue
            ratio = len(child1) / len(y)
            qualities = np.append(qualities,
                                   parent_impurity - ratio * gini_index(child1) - (1 - ratio) * gini_index(child2))
            x_kept.append(point)
        if not list(qualities):
            continue
        candidate = np.max(qualities)
        ind = np.argmax(qualities)
        if candidate > best_quality:
            best_quality = candidate
            target = x_kept[ind]
            col_target = j
    return col_target, target


# Calculate the Gini index for a split dataset
def gini_index(labels):
    labels = list(labels)
    numerator = np.sum(labels)
    div = len(labels)
    return (numerator/div)*(1-numerator/div)


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

x = pima[:, :-1]
y = pima[:,-1]
tree = tree_grow(x, y, 20, 5 )
print_tree(tree)

pred = tree_pred(x, tree)
print(conf(y,pred))
print(accuracy_score(y,pred))