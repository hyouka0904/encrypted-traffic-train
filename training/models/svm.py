from sklearn.svm import LinearSVC

NAME = "svm"
DEFAULT_PARAMS = {
    "C": 1.0,
    "max_iter": 2000,
}

def build(params: dict):
    return LinearSVC(**params)