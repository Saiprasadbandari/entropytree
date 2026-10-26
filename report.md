# Decision Tree from Scratch (entropy + information gain)

- **GitHub:** https://github.com/Saiprasadbandari/entropytree
- **PyPI (bonus):** https://pypi.org/project/sai-entropytree/

## What I built

`sai-entropytree` (imported as `entropytree`) is a binary decision tree classifier written from scratch in pure Python: only the standard library (`math`, `collections`) is imported, with no NumPy, sklearn or any other external library.

- **Entropy:** `H(y) = -Σ p_i log2 p_i`
- **Information gain:** `IG = H(parent) - (n_L/n)·H(left) - (n_R/n)·H(right)`
- **Splitting:** for each feature, sort values and try the midpoint between each pair of distinct neighbours as a threshold (`x <= t` goes left). The split with the highest information gain wins. Growth is recursive.
- **Stopping:** node is pure, `max_depth` reached, fewer than `min_samples_split` samples, or best gain below `min_gain`.
- **Size hyperparameter:** `max_depth` (plus `min_samples_split`, `min_gain`).
- **Bonus – display:** `tree.display(feature_names)` / `export_text()` prints the tree with class counts per leaf.
- **Bonus – package:** `pyproject.toml` with a `src/` layout; builds sdist and wheel and passes `twine check`. Install with `pip install sai-entropytree`.

## Usage

```python
from entropytree import DecisionTree

X = [[2.0, 1.0], [3.0, 1.5], [6.0, 4.0], [7.0, 4.5], [8.0, 5.0]]
y = ["small", "small", "big", "big", "big"]
tree = DecisionTree(max_depth=2).fit(X, y)
tree.display(["length", "width"])
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

## Effect of tree size

(sklearn was used only outside the package, to load the sample datasets and as a reference to compare against.) (test accuracy, 70/30 split, random_state=0)

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
- Numeric features only; no pruning. Split search sorts each feature and slides the threshold with running class counts, which is fast enough for small datasets.
- Tests use the standard `unittest` module (entropy, information gain, fit and depth control): `PYTHONPATH=src python -m unittest discover -s tests`.
