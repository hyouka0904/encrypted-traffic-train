# rpi_ap_train

加密流量分類模型的訓練 pipeline，產出的 ONNX 模型供 Raspberry Pi AP 部署端使用。
[encrypted-traffic-deploy](https://github.com/hyouka0904/encrypted-traffic-deploy)

## 專案結構

```
rpi_ap_train/
├── configs/
│   └── default.yaml          # 模型選擇與參數設定
├── data/
│   ├── Scenario B-ARFF/      # ISCX-VPN 資料集（不進 git）
│   └── processed/            # preprocess.py 的輸出（不進 git）
│       ├── train.csv
│       ├── test.csv
│       └── features.txt
├── models/                   # main.py 的輸出（不進 git）
│   ├── rf.onnx
│   ├── knn.onnx
│   ├── features.txt          # 自動複製，deploy 端需要
│   └── <model>_results.json  # 準確度 + 模型大小
├── training/
│   ├── models/
│   │   ├── rf.py             # Random Forest 定義
│   │   └── knn.py            # K-Nearest Neighbors 定義
│   ├── preprocess.py
│   ├── train.py              # fit / evaluate / export
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

## 使用方式

### Step 1 — Preprocess

```bash
python training/preprocess.py \
  --input "data/Scenario B-ARFF/TimeBasedFeatures-Dataset-15s-AllinOne.arff"
```

### Step 2 — 訓練與匯出 ONNX

編輯 `configs/default.yaml`，把 `model.name` 設成想要的模型，然後執行：

```bash
python training/main.py
# 或指定 config
python training/main.py --config configs/default.yaml
```

輸出：`models/<model>.onnx`、`models/features.txt`、`models/<model>_results.json`

## 支援的模型

| 名稱  | 演算法              | ONNX 匯出 |
|-------|---------------------|-----------|
| `rf`  | Random Forest       | ✅         |
| `knn` | K-Nearest Neighbors | ✅         |

### 新增模型

1. 在 `training/models/` 新增 `xxx.py`，實作 `NAME`、`DEFAULT_PARAMS`、`build(params)`
2. 在 `configs/default.yaml` 的 `model.params` 加對應的 key

---

## 下載預訓練模型（供 deploy 端使用）

訓練好的模型發布在 [GitHub Releases](https://github.com/hyouka0904/encrypted-traffic-train/releases)。

```bash
mkdir -p models

# 下載最新版
wget https://github.com/hyouka0904/encrypted-traffic-train/releases/latest/download/rf.onnx \
  -O models/rf.onnx

wget https://github.com/hyouka0904/encrypted-traffic-train/releases/latest/download/features.txt \
  -O models/features.txt
```

指定版本（例如 `v1.0-rf`）：

```bash
wget https://github.com/hyouka0904/encrypted-traffic-train/releases/download/v1.0-rf/rf.onnx \
  -O models/rf.onnx
```

### Release 命名規則

| tag        | 模型 | 備註          |
|------------|------|---------------|
| `v1.0-rf`  | RF   | baseline      |
| `v1.0-knn` | KNN  | n_neighbors=5 |

### 發布 Release
github CLI, linux:
```bash
type -p curl >/dev/null || sudo apt install curl -y

curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
| sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg

sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
| sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null

sudo apt update
sudo apt install gh -y
```
安裝完：

```bash
gh auth login
```

它會問：

```text
GitHub.com
HTTPS
Login with browser
```

之後：

1. terminal 會給你一個 code
2. 瀏覽器登入 GitHub
3. 貼上 code
4. authorize

完成後測試：

```bash
gh auth status
```

```bash
gh release create v1.X-<model_name> \
  models/<model_name>.onnx \
  models/features.txt \
  --title "<model_name> v1.X" \
```

## 待辦

- [ ] 新增 SVM、Naive Bayes（sklearn，可直接 ONNX 匯出）
- [ ] 評估深度學習模型：ResNet、LSTM、Transformer、DLinear，需 PyTorch，另行規劃
- [ ] train.py 以 isinstance 判斷 DL / sklearn 路徑，ONNX 輸出統一 label 格式，deploy 端不需另外處理