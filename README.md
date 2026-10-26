# entropytree

A decision tree classifier written from scratch (NumPy only) using Shannon entropy and information gain.

```bash
pip install sai-entropytree
```

```python
from sklearn.datasets import load_iris
from entropytree import DecisionTree

iris = load_iris()
tree = DecisionTree(max_depth=2).fit(iris.data, iris.target)
print(tree.score(iris.data, iris.target))
tree.display(iris.feature_names)
```

## Hyperparameters

- `max_depth` – size of the tree (default `None` = grow until pure)
- `min_samples_split` – minimum samples needed to split a node
- `min_gain` – minimum information gain to accept a split

## API

`fit(X, y)`, `predict(X)`, `score(X, y)`, `get_depth()`, `export_text(feature_names)`, `display(feature_names)`
