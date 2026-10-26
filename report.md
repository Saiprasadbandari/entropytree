# Decision Tree from Scratch (entropy + information gain)

- **GitHub:** https://github.com/Saiprasadbandari/entropytree
- **PyPI (bonus):** https://pypi.org/project/sai-entropytree/

## What I built

`sai-entropytree` (imported as `entropytree`) is a binary decision tree classifier written from scratch with NumPy only (no sklearn inside the package).

- **Entropy:** `H(y) = -Σ p_i log2 p_i`
- **Information gain:** `IG = H(parent) - (n_L/n)·H(left) - (n_R/n)·H(right)`
- **Splitting:** for each feature, sort values and try the midpoint between each pair of distinct neighbours as a threshold (`x <= t` goes left). The split with the highest information gain wins. Growth is recursive.
- **Stopping:** node is pure, `max_depth` reached, fewer than `min_samples_split` samples, or best gain below `min_gain`.
- **Size hyperparameter:** `max_depth` (plus `min_samples_split`, `min_gain`).
- **Bonus – display:** `tree.display(feature_names)` / `export_text()` prints the tree with class counts per leaf.
- **Bonus – package:** `pyproject.toml` with a `src/` layout; builds sdist and wheel and passes `twine check`. Install with `pip install sai-entropytree`.

## Usage

```python
from sklearn.datasets import load_iris
from entropytree import DecisionTree
iris = load_iris()
tree = DecisionTree(max_depth=2).fit(iris.data, iris.target)
tree.display(iris.feature_names)
```

```
|--- petal length (cm) <= 2.450
|   |--- class: 0  {0: 50}
|--- petal length (cm) >  2.450
|   |--- petal width (cm) <= 1.750
|   |   |--- class: 1  {1: 49, 2: 5}
|   |--- petal width (cm) >  1.750
|   |   |--- class: 2  {1: 1, 2: 45}
```

## Effect of tree size (test accuracy, 70/30 split, random_state=0)

| max_depth | iris (mine) | iris (sklearn, entropy) | breast cancer (mine) | breast cancer (sklearn) |
|---|---|---|---|---|
| 1 | 0.600 | 0.600 | 0.895 | 0.895 |
| 2 | 0.911 | 0.911 | 0.906 | 0.912 |
| 3 | 0.978 | 0.978 | 0.936 | 0.953 |
| 5 | 0.978 | 0.978 | 0.965 | 0.936 |
| None | 0.978 | 0.978 | 0.971 | 0.936 |

Very shallow trees underfit (iris depth 1: 0.60). Accuracy rises quickly, then plateaus. Results match sklearn on iris. On breast cancer the two differ by a few points because tie-breaking and threshold choice differ slightly.

## Limitations

- Greedy splitting cannot learn XOR-like patterns (every first split has zero gain).
- Numeric features only; no pruning; O(n²·d) split search, fine for small datasets.
- Tests: `pytest` (entropy, information gain, fit and depth control).
