import numpy as np
from collections import namedtuple

Leaf = namedtuple("Leaf", ("value",))
Node = namedtuple("Node", ("feature", "value", "left", "right"))

class BaseDecisionTree:
    def __init__(self, max_depth=np.inf, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        self.root = self._build_tree(X, y)
        return self

    def partition(self, X, y, feature, value):
        right = X[:, feature] >= value
        left = np.logical_not(right)
        return (X[left], y[left]), (X[right], y[right])

    def _build_tree(self, X, y, depth=1):
        if depth > self.max_depth or X.shape[0] < 2:
            return Leaf(self.leaf_value(y))
        feature, value = self.find_split(X, y)
        xy_left, xy_right = self.partition(X, y, feature, value)
        left = self._build_tree(*xy_left, depth=depth+1)
        right = self._build_tree(*xy_right, depth=depth+1)
        return Node(feature, value, left, right)

    def find_split(self, X, y):
        n_features = X.shape[1]
        max_features = self.max_features if self.max_features else n_features
        features = np.random.choice(n_features, max_features, replace=False)
        best_feature = None
        best_value = None
        best_impurity = np.inf
        for feature in features:
            order = np.argsort(X[:, feature])
            impurities = np.empty(order.shape[0] - 1)
            for split_index in np.arange(1, order.shape[0]):
                left = order[:split_index]
                right = order[split_index:]
                impurities[split_index - 1] = self.impurity(y[left], y[right])
            current_impurity = np.min(impurities)
            if current_impurity < best_impurity:
                best_impurity = current_impurity
                best_index = np.argmin(impurities) + 1
                best_feature = feature
                best_value = X[order[best_index], feature]
        return best_feature, best_value

    def impurity(self, y_left, y_right):
        left_count = y_left.shape[0]
        right_count = y_right.shape[0]
        count = left_count + right_count
        left_c = self.criteria(y_left)
        right_c = self.criteria(y_right)
        return (left_count / count * left_c + right_count / count * right_c)

    def _predict_one(self, x):
        node = self.root
        while not isinstance(node, Leaf):
            if x[node.feature] >= node.value:
                node = node.right
            else:
                node = node.left
        return node.value

    def predict(self, X):
        X = np.atleast_2d(X)
        y = np.empty(X.shape[0])
        for i, x in enumerate(X):
            y[i] = self._predict_one(x)
        return y

class DecisionTreeRegressor(BaseDecisionTree):
    def criteria(self, y):
        return np.var(y)

    def leaf_value(self, y):
        if len(y) == 0:
            return 0  
        return np.mean(y)