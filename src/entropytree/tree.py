"""Decision tree classifier built from scratch with entropy and information gain.

Pure Python: only the standard library (math, collections) is used.
"""
import math
from collections import Counter


def _entropy_from_counts(counts, total):
    """Entropy from a {class: count} mapping, given the total number of samples."""
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def entropy(labels):
    """Shannon entropy (base 2) of a list of labels: H = -sum(p * log2(p))."""
    if len(labels) == 0:
        return 0.0
    return _entropy_from_counts(Counter(labels), len(labels))


def information_gain(parent, left, right):
    """Entropy of the parent minus the weighted entropy of the two children."""
    n = len(parent)
    weighted = (len(left) / n) * entropy(left) + (len(right) / n) * entropy(right)
    return entropy(parent) - weighted


class _Node:
    def __init__(self, prediction, counts):
        self.prediction = prediction   # majority class at this node
        self.counts = counts           # {class: count} of the samples here
        self.feature = None            # feature index used to split (None = leaf)
        self.threshold = None          # go left if x[feature] <= threshold
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None


class DecisionTree:
    """Binary decision tree for numeric features.

    Parameters
    ----------
    max_depth : int or None
        Size hyperparameter: maximum depth of the tree (None = no limit).
    min_samples_split : int
        A node with fewer samples than this is not split.
    min_gain : float
        A split must improve information gain by at least this much.
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-12):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain

    # ---- training -------------------------------------------------------
    def fit(self, X, y):
        X = [list(row) for row in X]
        # Plain Python labels (also turns e.g. numpy scalars into int/str for clean printing).
        y = [v.item() if hasattr(v, "item") else v for v in y]
        self.n_features_ = len(X[0])
        self.root_ = self._grow(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        """Try every feature and every threshold; return the best by information gain."""
        n = len(y)
        parent_entropy = entropy(y)
        best_gain, best_feature, best_threshold = 0.0, None, None

        for f in range(self.n_features_):
            # Sort samples by this feature, then slide the split point left to right,
            # moving one sample at a time from the right side to the left side.
            order = sorted(range(n), key=lambda i: X[i][f])
            left_counts = Counter()
            right_counts = Counter(y)

            for k in range(1, n):
                label = y[order[k - 1]]
                left_counts[label] += 1
                right_counts[label] -= 1

                lo = X[order[k - 1]][f]
                hi = X[order[k]][f]
                if lo == hi:
                    continue  # cannot put a threshold between equal values

                child_entropy = (k / n) * _entropy_from_counts(left_counts, k) \
                    + ((n - k) / n) * _entropy_from_counts(right_counts, n - k)
                gain = parent_entropy - child_entropy

                if gain > best_gain:
                    best_gain, best_feature = gain, f
                    best_threshold = (lo + hi) / 2

        return best_gain, best_feature, best_threshold

    def _grow(self, X, y, depth):
        counts = dict(Counter(y))
        node = _Node(max(counts, key=counts.get), counts)

        # Stopping rules: pure node, too small, or tree is deep enough.
        if len(counts) == 1 or len(y) < self.min_samples_split:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node

        gain, feature, threshold = self._best_split(X, y)
        if feature is None or gain < self.min_gain:
            return node

        left_X, left_y, right_X, right_y = [], [], [], []
        for row, label in zip(X, y):
            if row[feature] <= threshold:
                left_X.append(row)
                left_y.append(label)
            else:
                right_X.append(row)
                right_y.append(label)

        node.feature, node.threshold = feature, threshold
        node.left = self._grow(left_X, left_y, depth + 1)
        node.right = self._grow(right_X, right_y, depth + 1)
        return node

    # ---- prediction -----------------------------------------------------
    def predict(self, X):
        return [self._predict_one(list(row)) for row in X]

    def _predict_one(self, row):
        node = self.root_
        while not node.is_leaf():
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.prediction

    def score(self, X, y):
        """Accuracy."""
        predictions = self.predict(X)
        y = list(y)
        return sum(p == t for p, t in zip(predictions, y)) / len(y)

    def get_depth(self):
        def depth(node):
            if node.is_leaf():
                return 0
            return 1 + max(depth(node.left), depth(node.right))
        return depth(self.root_)

    # ---- display --------------------------------------------------------
    def export_text(self, feature_names=None):
        """Return the tree as an indented text string."""
        names = feature_names or [f"x{i}" for i in range(self.n_features_)]
        lines = []

        def walk(node, indent):
            pad = "|   " * indent
            if node.is_leaf():
                lines.append(f"{pad}|--- class: {node.prediction}  {node.counts}")
                return
            name = names[node.feature]
            lines.append(f"{pad}|--- {name} <= {node.threshold:.3f}")
            walk(node.left, indent + 1)
            lines.append(f"{pad}|--- {name} >  {node.threshold:.3f}")
            walk(node.right, indent + 1)

        walk(self.root_, 0)
        return "\n".join(lines)

    def display(self, feature_names=None):
        print(self.export_text(feature_names))
