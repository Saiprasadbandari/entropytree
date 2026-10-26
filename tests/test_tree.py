import unittest
from entropytree import DecisionTree, entropy, information_gain


class TestTree(unittest.TestCase):
    def test_entropy(self):
        self.assertEqual(entropy([0, 0, 0]), 0)
        self.assertAlmostEqual(entropy([0, 1]), 1.0)

    def test_information_gain_perfect_split(self):
        y = [0, 0, 1, 1]
        self.assertAlmostEqual(information_gain(y, y[:2], y[2:]), 1.0)

    def test_fit_and_depth(self):
        X = [[1], [2], [3], [10], [11], [12], [20], [21]]
        y = [0, 0, 0, 1, 1, 1, 0, 0]
        self.assertEqual(DecisionTree().fit(X, y).score(X, y), 1.0)
        self.assertEqual(DecisionTree(max_depth=1).fit(X, y).get_depth(), 1)
        self.assertEqual(DecisionTree(max_depth=0).fit(X, y).get_depth(), 0)


if __name__ == "__main__":
    unittest.main()
