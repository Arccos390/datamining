import numpy as np


# Calculate the Gini index for a split dataset
def gini_index(dataset, target):
    total_of_group = len(dataset)

    count_for_target_1 = np.count_nonzero([dataset[:, 0] <= target])
    target_classes_1 = dataset[dataset[:, 0] <= target, -1]
    if len(target_classes_1) == 0:
        return None
    proportion_1 = np.count_nonzero(target_classes_1) / len(target_classes_1)
    score_1 = 1 - (proportion_1 ** 2 + (1 - proportion_1) ** 2)

    count_for_target_2 = np.count_nonzero([dataset[:, 0] > target])
    target_classes_2 = dataset[dataset[:, 0] > target, -1]
    if len(target_classes_2) == 0:
        return None
    proportion_2 = np.count_nonzero(target_classes_2) / len(target_classes_2)
    score_2 = 1 - (proportion_2 ** 2 + (1 - proportion_2) ** 2)

    gini = score_1 * count_for_target_1 / total_of_group + score_2 * count_for_target_2 / total_of_group
    return gini


# test Gini values
# credit_data = [
#     [22, 0, 0, 28, 1, 0],
#     [46, 0, 1, 32, 0, 0],
#     [24, 1, 1, 24, 1, 0],
#     [25, 0, 0, 27, 1, 0],
#     [29, 1, 1, 32, 0, 0],
#     [45, 1, 1, 30, 0, 1],
#     [63, 1, 1, 58, 1, 1],
#     [36, 1, 0, 52, 1, 1],
#     [23, 0, 1, 40, 0, 1],
#     [50, 1, 1, 28, 0, 1]
# ]
#

dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]


print(gini_index(np.array(dataset), 2.771244718))
