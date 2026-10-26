import numpy as np
from entropytree import DecisionTree, entropy, information_gain


def test_entropy():
    assert entropy(np.array([0, 0, 0])) == 0
    assert abs(entropy(np.array([0, 1])) - 1.0) < 1e-12


def test_information_gain_perfect_split():
    y = np.array([0, 0, 1, 1])
    assert abs(information_gain(y, y[:2], y[2:]) - 1.0) < 1e-12


def test_fit_and_depth():
    X = [[1], [2], [3], [10], [11], [12], [20], [21]]
    y = [0, 0, 0, 1, 1, 1, 0, 0]
    assert DecisionTree().fit(X, y).score(X, y) == 1.0
    assert DecisionTree(max_depth=1).fit(X, y).get_depth() == 1
    assert DecisionTree(max_depth=0).fit(X, y).get_depth() == 0
