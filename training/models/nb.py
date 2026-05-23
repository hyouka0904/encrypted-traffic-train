from sklearn.naive_bayes import GaussianNB

NAME = "nb"
DEFAULT_PARAMS = {
    "var_smoothing": 1e-9,
}

def build(params: dict):
    return GaussianNB(**params)