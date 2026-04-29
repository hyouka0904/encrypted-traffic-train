# Project Structure
目前只有做random forest, 之後會開yaml來選參數和模型還有輸出檔名
```
rpi_ap_train/
├── data/
│   ├── Scenario A1-ARFF/   # ISCX-VPN dataset (not tracked by git)
│   ├── Scenario A2-ARFF/
│   ├── Scenario B-ARFF/
│   └── processed/          # Output of preprocess.py (not tracked by git)
│       ├── train.csv
│       ├── test.csv
│       └── features.txt
├── training/
│   ├── preprocess.py
│   ├── train.py
│   └── model.onnx          # Output of train.py (not tracked by git)
├── requirements.txt
└── README.md
```

## Dataset

Download from [ISCX-VPN-NonVPN 2016](https://www.unb.ca/cic/datasets/vpn.html).  
Only **Scenario B-ARFF** is used for training (7-class application type classification).

Place the unzipped folders under `data/`.

## Setup

```bash
conda create -n changeme python=3.10
pip install -r requirements.txt
```

## Usage

**Step 1 — Preprocess**
```bash
python training/preprocess.py --input "data/Scenario B-ARFF/TimeBasedFeatures-Dataset-15s-AllinOne.arff"
```

**Step 2 — Train & Export ONNX**
```bash
python training/train.py
```

Output: `training/model.onnx`  


## Model

| Item | Detail |
|------|--------|
| Algorithm | Random Forest (n_estimators=100) |
| Features | 23 flow-level features (IAT, byte counts, duration, etc.) |
| Classes | VOIP / STREAMING / BROWSING / CHAT / MAIL / FT / P2P |
| macro F1 | 0.862 |
| Export format | ONNX opset 17 |