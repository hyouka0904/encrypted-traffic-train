import torch
import torch.nn as nn

from onnx_utils import export_dl


NAME = "cnn1d"
IS_DL = True

DEFAULT_PARAMS = {
    "channels": [32, 64, 128],
    "kernel_size": 5,
    "dropout": 0.3,
    "hidden_dim": 128,
}

TRAIN_PARAMS = {
    "epochs": 50,
    "lr": 1e-3,
    "batch_size": 256,
}


class CNN1D(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        channels,
        kernel_size: int,
        dropout: float,
        hidden_dim: int,
    ):
        super().__init__()

        padding = kernel_size // 2
        conv_layers = []
        in_channels = 1

        for out_channels in channels:
            conv_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*conv_layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        # Input shape from trainer: [batch_size, n_features]
        # CNN1D expects: [batch_size, channels, sequence_length]
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


def build(params: dict, n_features: int, n_classes: int) -> nn.Module:
    return CNN1D(
        n_features=n_features,
        n_classes=n_classes,
        channels=params.get("channels", [32, 64, 128]),
        kernel_size=params.get("kernel_size", 5),
        dropout=params.get("dropout", 0.3),
        hidden_dim=params.get("hidden_dim", 128),
    )


def export_onnx(model, feature_cols, output_path, **kwargs):
    label_encoder = kwargs.get("label_encoder")
    if label_encoder is None:
        raise ValueError("label_encoder is required for DL ONNX export")

    return export_dl(model, feature_cols, output_path, label_encoder)