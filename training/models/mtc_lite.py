import torch
import torch.nn as nn


NAME = "mtc_lite"
IS_DL = True

DEFAULT_PARAMS = {
    "max_packets": 32,
    "num_packet_features": 3,
    "hidden_dim": 64,
    "cnn_channels": 64,
    "transformer_layers": 2,
    "transformer_heads": 4,
    "dropout": 0.2,
}

TRAIN_PARAMS = {
    "epochs": 50,
    "lr": 1e-3,
    "batch_size": 128,
}


class MTCLite(nn.Module):
    """Dual-branch sequence model: 1D-CNN + Transformer over packet sequences.

    Expects input shape [batch_size, max_packets, num_features].
    """

    def __init__(
        self,
        num_packet_features: int,
        n_classes: int,
        max_packets: int,
        hidden_dim: int,
        cnn_channels: int,
        transformer_layers: int,
        transformer_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.max_packets = max_packets

        self.input_proj = nn.Linear(num_packet_features, hidden_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(cnn_channels + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, max_packets, num_features]
        x = self.input_proj(x)  # [batch_size, max_packets, hidden_dim]

        cnn_in = x.transpose(1, 2)  # [batch_size, hidden_dim, max_packets]
        cnn_out = self.cnn(cnn_in).squeeze(-1)  # [batch_size, cnn_channels]

        trans_out = self.transformer(x)  # [batch_size, max_packets, hidden_dim]
        trans_out = trans_out.mean(dim=1)  # [batch_size, hidden_dim]

        fused = torch.cat([cnn_out, trans_out], dim=1)
        return self.classifier(fused)


def build(params: dict, n_features: int, n_classes: int) -> nn.Module:
    num_packet_features = params.get("num_packet_features", n_features)
    return MTCLite(
        num_packet_features=num_packet_features,
        n_classes=n_classes,
        max_packets=params.get("max_packets", 32),
        hidden_dim=params.get("hidden_dim", 64),
        cnn_channels=params.get("cnn_channels", 64),
        transformer_layers=params.get("transformer_layers", 2),
        transformer_heads=params.get("transformer_heads", 4),
        dropout=params.get("dropout", 0.2),
    )


def export_onnx(model, feature_cols, output_path, **kwargs):
    raise NotImplementedError(
        "MTC-lite ONNX export is not implemented yet. "
        "Sequence models require sequence-aware dummy input "
        f"(batch, max_packets, num_features) and a dedicated training entry point; "
        "the current ARFF main.py path expects 2D tabular input."
    )
