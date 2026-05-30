# rpi_ap_train

加密流量分類模型的訓練 pipeline，產出的 ONNX 模型供 Raspberry Pi AP 部署端使用。
[encrypted-traffic-deploy](https://github.com/hyouka0904/encrypted-traffic-deploy)

## 專案結構

```
rpi_ap_train/
├── configs/
│   ├── default.yaml          # 範本（結構說明用）
│   ├── rf.yaml
│   ├── knn.yaml
│   ├── svm.yaml
│   ├── nb.yaml
│   ├── mlp.yaml
│   ├── cnn1d.yaml
│   ├── fttransformer.yaml
│   ├── xgb.yaml
│   └── lgb.yaml
├── data/
│   ├── Scenario B-ARFF/      # ISCX-VPN 資料集（不進 git）
│   └── processed/            # preprocess.py 的輸出（不進 git）
│       ├── train.csv
│       ├── test.csv
│       └── features.txt
├── models/                   # main.py 的輸出（不進 git）
│   ├── <model>.onnx
│   ├── features.txt          # 自動複製，deploy 端需要
│   ├── label_classes.txt     # mlp / cnn1d / xgb / lgb 專用，deploy 端查表用
│   └── <model>_results.json  # 準確度 + 模型大小
├── training/
│   ├── models/
│   │   ├── rf.py             # Random Forest
│   │   ├── knn.py            # K-Nearest Neighbors
│   │   ├── svm.py            # LinearSVC（含 StandardScaler Pipeline）
│   │   ├── nb.py             # Gaussian Naive Bayes
│   │   ├── mlp.py            # MLP（DL）
│   │   ├── cnn1d.py          # 1D CNN（DL）
│   │   ├── fttransformer.py  # FT-Transformer tabular DL
│   │   ├── xgb.py            # XGBoost
│   │   └── lgb.py            # LightGBM
│   ├── onnx_utils.py         # 共用 ONNX export 工具
│   ├── trainer.py            # fit / evaluate
│   ├── preprocess.py
│   └── main.py               # entry point
├── requirements.txt
└── README.md
```

## 資料集

從 [ISCX-VPN-NonVPN 2016](https://www.unb.ca/cic/datasets/vpn.html) 下載，使用 **Scenario B-ARFF**。

解壓後放在 `data/` 底下。

## 安裝

```bash
conda create -n rpi-train python=3.10
pip install -r requirements.txt
```

### PyTorch（MLP 使用）

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

CPU only：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 使用方式

### Step 1 — Preprocess

```bash
python training/preprocess.py --input "data/Scenario B-ARFF/TimeBasedFeatures-Dataset-15s-AllinOne.arff"
```

### Step 2 — 訓練與匯出 ONNX

```bash
python training/main.py --config <yaml_path>
```

DL 可指定訓練裝置（預設 `auto`，有 GPU 就用）：

```bash
python training/main.py --config configs/<your_DL_model>.yaml --device cpu
python training/main.py --config configs/<your_DL_model>.yaml --device cuda
```

輸出：`models/<model>.onnx`、`models/features.txt`、`models/<model>_results.json`

mlp / cnn1d / xgb / lgb 額外輸出：`models/label_classes.txt`

### Config 格式

sklearn 模型（參考 `configs/default.yaml`）：

```yaml
model:
  name: rf
  params:
    n_estimators: 100

data:
  processed_dir: data/processed

output:
  dir: models
```

DL 模型（額外支援 `train_params`）：

```yaml
model:
  name: mlp
  params:
    hidden_dims: [256, 128, 64]
    dropout: 0.3
  train_params:
    epochs: 50
    lr: 0.001
    batch_size: 256

data:
  processed_dir: data/processed

output:
  dir: models
```

## 支援的模型

| 名稱 | 演算法 | 類型 | macro F1 | ONNX 匯出 | 備註 |
|---|---|---|---:|---|---|
| `xgb` | XGBoost | sklearn | 0.8865 | ✅ | Best current deployment model |
| `lgb` | LightGBM | sklearn | 0.8807 | ✅ | Close to XGBoost but larger ONNX |
| `rf` | Random Forest | sklearn | 0.862 | ✅ | Strong baseline |
| `mlp` | Multi-Layer Perceptron | DL | 0.5938 | ✅ | Best DL baseline, smaller but weaker |
| `cnn1d` | Lightweight 1D CNN | DL | 0.4260 | ✅ | Paper-inspired experimental model |
| `fttransformer` | FT-Transformer (tabular) | DL | — | ✅ | Tabular Transformer experiment; no result yet |
| `knn` | K-Nearest Neighbors | sklearn | — | ✅ | baseline |
| `svm` | LinearSVC + StandardScaler | sklearn | — | ✅ | Pipeline 自動處理特徵縮放 |
| `nb` | Gaussian Naive Bayes | sklearn | — | ✅ | baseline 參考用，準確度較低 |

### Experiment conclusion

Although CNN1D and MLP were tested as deep learning models, the current dataset is based on tabular flow features rather than raw packet sequences. In this setting, tree-based models perform much better. XGBoost achieved the best macro F1 and accuracy, so it is selected as the current deployment model for Raspberry Pi AP inference.

### XGBoost tuning experiment

The original XGBoost configuration remains the best deployment model.

| Config | Macro F1 | Accuracy | ONNX size | Decision |
|---|---:|---:|---:|---|
| `xgb.yaml` | 0.8865 | 0.9120 | 3964.9 KB | Selected |
| `xgb_n200_d6_lr010` | 0.8814 | 0.9078 | 2828.6 KB | Smaller, but rejected due to F1 drop |

Although `xgb_n200_d6_lr010` reduces ONNX size by about 29%, its macro F1 drops by 0.0051, so the baseline `xgb.yaml` is still selected for deployment.

### Next research direction: MTC-lite

XGBoost (`configs/xgb.yaml`, macro F1 **0.8865**) remains the **selected deployment model**. No change to that baseline.

MTC-lite is a **future raw-packet sequence experiment**, inspired by paper models such as MTC (Transformer + 1D-CNN). Tabular DL models (MLP, CNN1D) on the 23-feature ARFF dataset did not beat XGBoost because they lack packet-level sequence input. MTC-lite is **not a replacement yet** — there are no reported results.

Scaffold (research only):

| Path | Purpose |
|------|---------|
| `docs/mtc_lite_research_plan.md` | Research plan and evaluation criteria |
| `training/sequence/` | Future PCAP → NPZ preprocessing |
| `training/models/mtc_lite.py` | PyTorch dual-branch model definition |
| `configs/mtc_lite.yaml` | Future config (not wired to ARFF `main.py`) |

Target: beat XGBoost macro F1 **0.8865** using packet sequences before any deployment switch.

### FT-Transformer (tabular DL experiment)

`fttransformer` is a paper-inspired **FT-Transformer-style** model for the existing ARFF tabular features (`data/processed/`). Unlike MTC-lite, it does **not** require raw PCAP sequences and runs through the standard `training/main.py` pipeline.

```bash
python training/main.py --config configs/fttransformer.yaml
```

No benchmark result is reported yet. XGBoost (`configs/xgb.yaml`, macro F1 **0.8865**) remains the selected deployment model until `fttransformer` beats that baseline on a fair evaluation.

### 新增模型

每個模型檔案放在 `training/models/` 下，需實作以下欄位：

**共通（所有模型）：**

```python
NAME = "xxx"
DEFAULT_PARAMS = { ... }

def build(params: dict):
    ...

def export_onnx(model, feature_cols, output_path, **kwargs):
    # sklearn-like: 使用 onnx_utils.export_sklearn
    # DL:          使用 onnx_utils.export_dl
    # 自訂:        自行實作
    ...
```

**DL 模型額外需要：**

```python
IS_DL = True

TRAIN_PARAMS = {
    "epochs": 50,
    "lr": 1e-3,
    "batch_size": 256,
}

def build(params: dict, n_features: int, n_classes: int) -> nn.Module:
    ...
```

新增 `configs/xxx.yaml` 後即可直接使用。

---

## 下載預訓練模型（供 deploy 端使用）

訓練好的模型發布在 [GitHub Releases](https://github.com/hyouka0904/encrypted-traffic-train/releases)。

```bash
mkdir -p models
```

**rf / knn / svm / nb**（只需 onnx + features.txt）：

```bash
wget https://github.com/hyouka0904/encrypted-traffic-train/releases/latest/download/rf.onnx -O models/rf.onnx
wget https://github.com/hyouka0904/encrypted-traffic-train/releases/latest/download/features.txt -O models/features.txt
```

**mlp / cnn1d / xgb / lgb**（需額外下載 label_classes.txt）：

```bash
wget https://github.com/hyouka0904/encrypted-traffic-train/releases/latest/download/xgb.onnx -O models/xgb.onnx
wget https://github.com/hyouka0904/encrypted-traffic-train/releases/latest/download/features.txt -O models/features.txt
wget https://github.com/hyouka0904/encrypted-traffic-train/releases/latest/download/label_classes.txt -O models/label_classes.txt
```

指定版本（例如 `v1.0-xgb`）：

```bash
wget https://github.com/hyouka0904/encrypted-traffic-train/releases/download/v1.0-xgb/xgb.onnx -O models/xgb.onnx
```

### 發布 Release（GitHub CLI）

**安裝（Linux）：**

```bash
type -p curl >/dev/null || sudo apt install curl -y
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh -y
```

**安裝（Windows）：**

下載 `gh_*_windows_amd64.msi` 並執行：https://github.com/cli/cli/releases/latest

**登入：**

```bash
gh auth login
# 選 GitHub.com → HTTPS → Login with a web browser
```

**發布：**

rf / knn / svm / nb（不需 label_classes.txt）：

```bash
gh release create v1.X-<model_name> models/<model_name>.onnx models/features.txt --title "<model_name> v1.X"
```

mlp / cnn1d / xgb / lgb（需帶 label_classes.txt）：

```bash
gh release create v1.X-<model_name> models/<model_name>.onnx models/features.txt models/label_classes.txt --title "<model_name> v1.X"
```

## 待辦

- [ ] study paper and implement more models