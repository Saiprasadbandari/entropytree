# entropytree

A decision tree classifier written from scratch in pure Python (standard library only, no NumPy or other dependencies) using Shannon entropy and information gain.

```bash
pip install sai-entropytree
```

```python
from entropytree import DecisionTree

X = [[2.0, 1.0], [3.0, 1.5], [6.0, 4.0], [7.0, 4.5], [8.0, 5.0]]
y = ["small", "small", "big", "big", "big"]

tree = DecisionTree(max_depth=2).fit(X, y)
print(tree.predict([[2.5, 1.2], [7.5, 4.8]]))   # ['small', 'big']
print(tree.score(X, y))                          # 1.0
tree.display(["length", "width"])
```

## Hyperparameters

- `max_depth` – size of the tree (default `None` = grow until pure)
- `min_samples_split` – minimum samples needed to split a node
- `min_gain` – minimum information gain to accept a split

## API

`fit(X, y)`, `predict(X)`, `score(X, y)`, `get_depth()`, `export_text(feature_names)`, `display(feature_names)`
