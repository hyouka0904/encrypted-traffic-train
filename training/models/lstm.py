import torch.nn as nn

NAME      = "lstm"
IS_DL     = True

DEFAULT_PARAMS = {
    "hidden_size": 128,
    "num_layers":  2,
    "dropout":     0.3,
}

TRAIN_PARAMS = {
    "epochs":     50,
    "lr":         1e-3,
    "batch_size": 256,
}


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # dropout 只在 num_layers > 1 時有效，否則 PyTorch 會印 warning
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: [batch, n_features]
        x = x.unsqueeze(1)          # → [batch, 1, n_features]（single timestep）
        out, _ = self.lstm(x)       # out: [batch, 1, hidden_size]
        out = out[:, -1, :]         # last timestep → [batch, hidden_size]
        return self.classifier(out) # → [batch, n_classes]


def build(params: dict, n_features: int, n_classes: int) -> LSTMClassifier:
    return LSTMClassifier(
        n_features=n_features,
        n_classes=n_classes,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    )