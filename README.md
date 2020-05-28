# LUCK

Perform LUCK clustering from vector array.

LUCK - Linear Correlation Clustering Using Cluster Algorithms and a kNN based Distance Function

Method which allows distance-based clustering algorithm to find linear correlated data.

## Example
```python
import numpy as np
from LUCK import LUCK, KMEANS
from sklearn.cluster import DBSCAN

if __name__ == '__main__':
    X = np.array([[0, 1], [1, 1.5], [2, 2], [3, 2.5], [4, 3], [5, 3.5], [6, 4],
                  [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],
                  [5, 1], [4.5, 1.5], [4, 2], [3.5, 2.5], [3, 3], [2.5, 3.5]])

    luck_kmeans = LUCK(threshold=0.04, algo=KMEANS(k=3)).fit(X)
    print(luck_kmeans.labels)
    # [-2  0  0 -2  0  0  0  1  1  1  1  1 -2  2  2  2 -2  2]
    
    luck_dbscan = LUCK(threshold=0.04, algo=DBSCAN(eps=0.001, min_samples=3)).fit(X)
    print(luck_dbscan.labels)
    # [-2  0  0 -2  0  0  0 -1  1  1  1 -1 -2  2  2 -1 -2  2]
```

## Documentation
Link to the [documentation](./docs/index.html).

## Paper
Link to the [paper](https://dl.acm.org/doi/abs/10.1145/3335783.3335801) for more information about LUCK.

