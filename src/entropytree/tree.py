"""Decision tree classifier built from scratch with entropy and information gain."""
import numpy as np


def entropy(y):
    """Shannon entropy (base 2) of a label array."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def information_gain(y, y_left, y_right):
    """Entropy of the parent minus the weighted entropy of the two children."""
    n = len(y)
    child = len(y_left) / n * entropy(y_left) + len(y_right) / n * entropy(y_right)
    return entropy(y) - child


class _Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, counts=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value      # predicted class (set on every node)
        self.counts = counts    # class counts at this node

    @property
    def is_leaf(self):
        return self.left is None


class DecisionTree:
    """Binary decision tree for numeric features.

    Parameters
    ----------
    max_depth : int or None
        Size hyperparameter: maximum depth of the tree (None = unlimited).
    min_samples_split : int
        Minimum samples a node needs to be split.
    min_gain : float
        Minimum information gain required to make a split.
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-12):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.root_ = self._grow(X, y, 0)
        return self

    def _best_split(self, X, y):
        best_gain, best_feat, best_thr = 0.0, None, None
        n = len(y)
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f], kind="stable")
            xs, ys = X[order, f], y[order]
            for i in range(1, n):
                if xs[i] == xs[i - 1]:
                    continue
                gain = information_gain(ys, ys[:i], ys[i:])
                if gain > best_gain:
                    best_gain, best_feat = gain, f
                    best_thr = (xs[i] + xs[i - 1]) / 2
        return best_gain, best_feat, best_thr

    def _grow(self, X, y, depth):
        classes, counts = np.unique(y, return_counts=True)
        node = _Node(value=classes[np.argmax(counts)],
                     counts=dict(zip(classes.tolist(), counts.tolist())))
        if (len(classes) == 1
                or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node
        gain, feat, thr = self._best_split(X, y)
        if feat is None or gain < self.min_gain:
            return node
        mask = X[:, feat] <= thr
        node.feature, node.threshold = feat, thr
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def get_depth(self):
        def d(n):
            return 0 if n.is_leaf else 1 + max(d(n.left), d(n.right))
        return d(self.root_)

    def export_text(self, feature_names=None):
        """Return the tree as an indented text string."""
        names = feature_names or [f"x{i}" for i in range(self.n_features_)]
        lines = []

        def rec(n, indent):
            pad = "|   " * indent
            if n.is_leaf:
                lines.append(f"{pad}|--- class: {n.value}  {n.counts}")
                return
            lines.append(f"{pad}|--- {names[n.feature]} <= {n.threshold:.3f}")
            rec(n.left, indent + 1)
            lines.append(f"{pad}|--- {names[n.feature]} >  {n.threshold:.3f}")
            rec(n.right, indent + 1)

        rec(self.root_, 0)
        return "\n".join(lines)

    def display(self, feature_names=None):
        print(self.export_text(feature_names))
