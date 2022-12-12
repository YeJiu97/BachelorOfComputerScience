import sys
import numpy as np
from collections import Counter


def load_data(filename):
    with open(filename, 'r') as f:
        content = f.read().splitlines()
    content = [item.split() for item in content]
    # skip the header
    content = content[1:]
    for i in range(len(content)):
        content[i] = list(map(float, content[i]))
    data = np.array(content)
    return data


def equal_features(X):
    """
    Given the feature matrix, check if all the features are same
    :param X: n*d numpy matrix
    :return: True if all the features are same, False otherwise
    """
    mean = X.mean(axis=0)
    norms = np.linalg.norm(X - mean, axis=1)
    return np.max(norms) < 1e-10


def calculate_entropy(y):
    """
    Calculate the entropy
    :param y: 1*d vector, the labels
    :return: entropy
    """
    counter = Counter(y)
    for label in counter:
        counter[label] /= len(y)
    return sum(- p * np.log2(p) for label, p in counter.items())


def calculate_information_gain(split_val, x, y):
    """
    Calculate the information gain when split the dataset by feature x using the split val
    :param split_val: float, the split value
    :param x: 1*d vector, the feature vector
    :param y: 1*d vector, the labels
    :return: information gain
    """
    ent = calculate_entropy(y)
    # split
    left_indices = np.where(x <= split_val)[0]
    right_indices = np.where(x > split_val)[0]
    # calculate left/right entropy
    left_entropy = calculate_entropy(y[left_indices])
    right_entropy = calculate_entropy(y[right_indices])
    return ent - (len(left_indices) / len(y) * left_entropy + len(right_indices) / len(y) * right_entropy)


class DecisionTreeNode:
    def __init__(self, feature_idx=None, split_val=None, label=None, counter=None):
        """
        :param feature_idx: index of the feature to split the
        :param split_val: the split value
        :param counter: counter
        """
        self.feature_idx = feature_idx
        self.split_val = split_val
        self.label = label
        self.counter = counter

        # left child and right child
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None


def choose_split(X, y):
    """
    Given the features and labels, return the best feature to split the data
    :param X: n*d matrix
    :param y: 1*d vector, the labels
    :return: best attribute and the split vals
    """
    best_gain = 0
    best_attr = None
    best_split = None
    d = X.shape[1]
    for i in range(d):
        # get the value of i-th feature
        feature_values = X[:, i]
        # get the sorted unique values
        feature_values = sorted(set(feature_values))
        # if there is only 1 unique values, it's not splittable
        if len(feature_values) == 1:
            # print(X, y)
            # raise AssertionError('Feature {} is unique'.format(i))
            continue
        else:
            # try to split the dataset by all possible middle point
            for j in range(1, len(feature_values)):
                split_val = (feature_values[j] + feature_values[j - 1]) / 2
                gain = calculate_information_gain(split_val, X[:, i], y)
                # better
                if gain > best_gain:
                    best_gain = gain
                    best_split = split_val
                    best_attr = i
    return best_attr, best_split


def get_label(counter):
    mode = max(counter, key=counter.get)
    value_counts = Counter(counter.values())
    if value_counts[counter[mode]] > 1:
        return -1
    return mode


def dtl(X, y, min_leaf):
    """Algorithm 1"""
    n = len(X)
    if n <= min_leaf or len(set(y)) == 1 or equal_features(X):
        counter = Counter(y)
        label = get_label(counter)
        leaf = DecisionTreeNode(label=int(label), counter=counter)
        return leaf
    best_attr, best_split = choose_split(X, y)
    node = DecisionTreeNode(feature_idx=best_attr, split_val=best_split)
    # split
    left_indices = np.where(X[:, best_attr] <= best_split)[0]
    right_indices = np.where(X[:, best_attr] > best_split)[0]

    node.left = dtl(X[left_indices], y[left_indices], min_leaf)
    node.right = dtl(X[right_indices], y[right_indices], min_leaf)
    return node


def predict_dtl(tree, feature):
    """
    Prediction
    :param tree: the tree root
    :param feature: 1*d vector, the features
    :return: the predicted label
    """
    cur = tree
    while not cur.is_leaf():
        split_val, split_feature = cur.split_val, cur.feature_idx
        if feature[split_feature] <= split_val:
            cur = cur.left
        else:
            cur = cur.right
    return cur.label


def main():
    if len(sys.argv) != 4:
        print('Usage: python %s [train] [test] [minleaf]' % (sys.argv[0]))
        sys.exit(-1)
    train = sys.argv[1]
    test = sys.argv[2]
    min_leaf = int(sys.argv[3])

    train = load_data(train)
    X, y = train[:, :-1], train[:, -1]
    y = y.astype(int)
    tree = dtl(X, y, min_leaf)

    X_test = load_data(test)
    y_test = [predict_dtl(tree, X_test[i]) for i in range(len(X_test))]
    for e in y_test:
        print(e)

    # y_train_pred = [predict_dtl(tree, train[i]) for i in range(len(train))]
    # print(sum(y_train_pred[i] == y[i] for i in range(len(y_train_pred))))
    # print(len(y_train_pred ))


if __name__ == '__main__':
    main()
