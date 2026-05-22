from sklearn.ensemble import RandomForestClassifier

NAME = "rf"

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def build(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**params)