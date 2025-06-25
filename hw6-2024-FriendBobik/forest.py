import numpy as np
from tree import DecisionTreeRegressor

class RandomForestRegressor:
    def __init__(self, num_trees=100, tree_depth=np.inf):
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.trees = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.num_trees):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_subset = X[indices]
            y_subset = y[indices]

            tree = DecisionTreeRegressor(max_depth=self.tree_depth)
            tree.fit(X_subset, y_subset)
            self.trees.append(tree)

        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)