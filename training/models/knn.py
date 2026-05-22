from sklearn.neighbors import KNeighborsClassifier

NAME = "knn"

DEFAULT_PARAMS = {
    "n_neighbors": 5,
    "weights": "uniform",
    "metric": "minkowski",
    "n_jobs": -1,
}


def build(params: dict) -> KNeighborsClassifier:
    return KNeighborsClassifier(**params)