import torch
import torch.nn as nn

from onnx_utils import export_dl


NAME = "mlp"
IS_DL = True

DEFAULT_PARAMS = {
    "hidden_dims": [256, 128, 64],
    "dropout": 0.3,
    "batch_norm": True,
}

TRAIN_PARAMS = {
    "epochs": 50,
    "lr": 1e-3,
    "batch_size": 256,
}


class MLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dims, dropout: float, batch_norm: bool):
        super().__init__()

        layers = []
        in_dim = n_features

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build(params: dict, n_features: int, n_classes: int) -> nn.Module:
    return MLP(
        n_features=n_features,
        n_classes=n_classes,
        hidden_dims=params.get("hidden_dims", [256, 128, 64]),
        dropout=params.get("dropout", 0.3),
        batch_norm=params.get("batch_norm", True),
    )


def export_onnx(model, feature_cols, output_path, **kwargs):
    label_encoder = kwargs.get("label_encoder")
    if label_encoder is None:
        raise ValueError("label_encoder is required for DL ONNX export")

    return export_dl(model, feature_cols, output_path, label_encoder)