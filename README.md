# rpi_ap_train

加密流量分類模型的訓練 pipeline，產出的 ONNX 模型供 Raspberry Pi AP 部署端使用。
[encrypted-traffic-deploy](https://github.com/hyouka0904/encrypted-traffic-deploy)

## 專案結構

​```
rpi_ap_train/
├── configs/
│   ├── default.yaml          # 範本（結構說明用）
│   ├── rf.yaml
│   ├── knn.yaml
│   ├── svm.yaml
│   ├── nb.yaml
│   └── lstm.yaml
├── data/
│   ├── Scenario B-ARFF/      # ISCX-VPN 資料集（不進 git）
│   └── processed/            # preprocess.py 的輸出（不進 git）
│       ├── train.csv
│       ├── test.csv
│       └── features.txt
├── models/                   # main.py 的輸出（不進 git）
│   ├── <model>.onnx
│   ├── features.txt          # 自動複製，deploy 端需要
│   ├── label_classes.txt     # DL 模型專用，deploy 端 argmax 查表用
│   └── <model>_results.json  # 準確度 + 模型大小
├── training/
│   ├── models/
│   │   ├── rf.py             # Random Forest
│   │   ├── knn.py            # K-Nearest Neighbors
│   │   ├── svm.py            # LinearSVC（含 StandardScaler Pipeline）
│   │   ├── nb.py             # Gaussian Naive Bayes
│   │   └── lstm.py           # LSTM（DL）
│   ├── train_sklearn.py      # sklearn fit / evaluate / export
│   ├── train_dl.py           # PyTorch fit / evaluate / export
│   ├── preprocess.py
│   └── main.py               # entry point
├── requirements.txt
└── README.md
​```

## 資料集

從 [ISCX-VPN-NonVPN 2016](https://www.unb.ca/cic/datasets/vpn.html) 下載，使用 **Scenario B-ARFF**。

解壓後放在 `data/` 底下。

## 安裝

​```bash
conda create -n rpi-train python=3.10
pip install -r requirements.txt
​```

### PyTorch（DL 模型使用）

​```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
​```

CPU only：

​```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
​```

## 使用方式

### Step 1 — Preprocess

​```bash
python training/preprocess.py \
  --input "data/Scenario B-ARFF/TimeBasedFeatures-Dataset-15s-AllinOne.arff"
​```

### Step 2 — 訓練與匯出 ONNX

​```bash
python training/main.py --config configs/rf.yaml
python training/main.py --config configs/knn.yaml
python training/main.py --config configs/svm.yaml
python training/main.py --config configs/nb.yaml
python training/main.py --config configs/lstm.yaml
​```

DL 模型可指定訓練裝置（預設 `auto`，有 GPU 就用）：

​```bash
python training/main.py --config configs/lstm.yaml --device cpu
python training/main.py --config configs/lstm.yaml --device cuda
​```

輸出：`models/<model>.onnx`、`models/features.txt`、`models/<model>_results.json`

DL 模型額外輸出：`models/label_classes.txt`

### Config 格式

sklearn 模型（參考 `configs/default.yaml`）：

​```yaml
model:
  name: rf
  params:
    n_estimators: 100

data:
  processed_dir: data/processed

output:
  dir: models
​```

DL 模型（額外支援 `train_params`）：

​```yaml
model:
  name: lstm
  params:           # 模型結構參數
    hidden_size: 128
    num_layers: 2
    dropout: 0.3
  train_params:     # 訓練超參，蓋過 TRAIN_PARAMS
    epochs: 50
    lr: 0.001
    batch_size: 256

data:
  processed_dir: data/processed

output:
  dir: models
​```

## 支援的模型

| 名稱   | 演算法                     | 類型    | ONNX 匯出 | 備註                        |
|--------|----------------------------|---------|-----------|-----------------------------|
| `rf`   | Random Forest              | sklearn | ✅         |                             |
| `knn`  | K-Nearest Neighbors        | sklearn | ✅         |                             |
| `svm`  | LinearSVC + StandardScaler | sklearn | ✅         | Pipeline 自動處理特徵縮放   |
| `nb`   | Gaussian Naive Bayes       | sklearn | ✅         | baseline 參考用，準確度較低 |
| `lstm` | LSTM                       | DL      | ✅         | 需要 PyTorch                |

### 新增模型

**sklearn 模型：**

1. 在 `training/models/` 新增 `xxx.py`，實作 `NAME`、`DEFAULT_PARAMS`、`build(params)`
2. 新增 `configs/xxx.yaml`

**DL 模型：**

1. 在 `training/models/` 新增 `xxx.py`，實作以下欄位：

​```python
IS_DL = True                  # DL 模型必須宣告

NAME = "xxx"

DEFAULT_PARAMS = { ... }      # 模型結構參數

TRAIN_PARAMS = {              # 訓練超參預設值
    "epochs": 50,
    "lr": 1e-3,
    "batch_size": 256,
}

def build(params: dict, n_features: int, n_classes: int) -> nn.Module:
    ...
​```

2. 新增 `configs/xxx.yaml`

---

## 下載預訓練模型（供 deploy 端使用）

訓練好的模型發布在 [GitHub Releases](https://github.com/hyouka0904/encrypted-traffic-train/releases)。

​```bash
mkdir -p models

# sklearn 模型（以 rf 為例）
wget https://github.com/hyouka0904/encrypted-traffic-train/releases/latest/download/rf.onnx \
  -O models/rf.onnx

wget https://github.com/hyouka0904/encrypted-traffic-train/releases/latest/download/features.txt \
  -O models/features.txt

# DL 模型（以 lstm 為例，需額外下載 label_classes.txt）
wget https://github.com/hyouka0904/encrypted-traffic-train/releases/latest/download/lstm.onnx \
  -O models/lstm.onnx

wget https://github.com/hyouka0904/encrypted-traffic-train/releases/latest/download/label_classes.txt \
  -O models/label_classes.txt
​```

指定版本（例如 `v1.0-rf`）：

​```bash
wget https://github.com/hyouka0904/encrypted-traffic-train/releases/download/v1.0-rf/rf.onnx \
  -O models/rf.onnx
​```

### 發布 Release（GitHub CLI）

**安裝（Linux）：**

​```bash
type -p curl >/dev/null || sudo apt install curl -y

curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
| sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg

sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
| sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null

sudo apt update
sudo apt install gh -y
​```

**安裝（Windows）：**

下載 `gh_*_windows_amd64.msi` 並執行：https://github.com/cli/cli/releases/latest

**登入：**

​```bash
gh auth login
# 選 GitHub.com → HTTPS → Login with a web browser
​```

**發布：**

sklearn 模型：

​```bash
gh release create v1.X-<model_name> \
  models/<model_name>.onnx \
  models/features.txt \
  --title "<model_name> v1.X"
​```

DL 模型（多帶 `label_classes.txt`）：

​```bash
gh release create v1.X-<model_name> \
  models/<model_name>.onnx \
  models/features.txt \
  models/label_classes.txt \
  --title "<model_name> v1.X"
​```

## 待辦

- [ ] 評估深度學習模型：ResNet、Transformer、DLinear，需 PyTorch，另行規劃
- [ ] train_dl.py 以 isinstance 判斷 DL 路徑，ONNX 輸出統一 label 格式，deploy 端不需另外處理