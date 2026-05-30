import torch
import torch.nn as nn

from onnx_utils import export_dl


NAME = "fttransformer"
IS_DL = True

DEFAULT_PARAMS = {
    "hidden_dim": 64,
    "num_layers": 3,
    "num_heads": 4,
    "dropout": 0.2,
    "ff_dim": 128,
}

TRAIN_PARAMS = {
    "epochs": 80,
    "lr": 1e-3,
    "batch_size": 256,
}


class FTTransformer(nn.Module):
    """FT-Transformer-style model for numeric tabular features.

    Each scalar feature becomes one token via learnable per-feature affine maps.
    A CLS token aggregates the sequence for classification.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ff_dim: int,
    ):
        super().__init__()
        self.n_features = n_features

        self.weight = nn.Parameter(torch.empty(n_features, hidden_dim))
        self.bias = nn.Parameter(torch.empty(n_features, hidden_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, n_features]
        tokens = x.unsqueeze(-1) * self.weight + self.bias  # [B, n_features, hidden_dim]

        batch_size = tokens.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)

        encoded = self.transformer(seq)
        cls_out = encoded[:, 0, :]
        return self.head(cls_out)


def build(params: dict, n_features: int, n_classes: int) -> nn.Module:
    return FTTransformer(
        n_features=n_features,
        n_classes=n_classes,
        hidden_dim=params.get("hidden_dim", 64),
        num_layers=params.get("num_layers", 3),
        num_heads=params.get("num_heads", 4),
        dropout=params.get("dropout", 0.2),
        ff_dim=params.get("ff_dim", 128),
    )


def export_onnx(model, feature_cols, output_path, **kwargs):
    label_encoder = kwargs.get("label_encoder")
    if label_encoder is None:
        raise ValueError("label_encoder is required for DL ONNX export")

    return export_dl(model, feature_cols, output_path, label_encoder)
