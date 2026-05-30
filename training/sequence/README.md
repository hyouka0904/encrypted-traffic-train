# Packet sequence dataset format

This folder holds preprocessing code for **raw packet sequence** experiments (MTC-lite and successors).

The current tabular training pipeline still uses ARFF features via `training/preprocess.py` → `data/processed/`. Sequence data is a separate research path.

## NPZ format

Sequence datasets are stored as NumPy compressed archives (`.npz`).

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `X` | `float32` | `[num_flows, max_packets, num_features]` | Per-flow packet sequences (padded/truncated to `max_packets`) |
| `y` | string array | `[num_flows]` | Traffic category labels |
| `feature_names` | string array | — | `["packet_len", "direction", "iat"]` |
| `max_packets` | int | scalar | Truncation/padding length used during build |
| `classes` | string array | — | Known class names |

### Feature dimensions

Each packet in a flow is described by:

- **packet_len** — total packet length in bytes
- **direction** — `+1.0` or `-1.0` relative to the canonical flow direction
- **iat** — inter-arrival time since the previous packet in the flow (seconds; first packet = `0.0`)

Future versions may append normalized payload byte columns (`byte_0` … `byte_{N-1}`).

### Example output path

```
data/sequence/iscx_vpn_sequence.npz
```

## Build sequence dataset

`build_sequence_dataset.py` reads IPv4 TCP/UDP PCAPs, groups packets into bidirectional flows, and writes the NPZ file above.

Install dependency:

```bash
pip install scapy
```

### Filename label mode

Infers class from the PCAP filename or parent directory (case-insensitive). Example: `browsing_sample.pcap` → `BROWSING`.

```bash
python training/sequence/build_sequence_dataset.py \
  --pcap-dir data/raw_pcap \
  --output data/sequence/iscx_vpn_sequence.npz \
  --label-mode filename
```

### CSV label mode

Maps each PCAP basename to a label via CSV. PCAPs missing from the CSV are skipped with a warning.

Expected `labels.csv` format:

```csv
file,label
sample1.pcap,BROWSING
sample2.pcap,CHAT
```

```bash
python training/sequence/build_sequence_dataset.py \
  --pcap-dir data/raw_pcap \
  --labels data/sequence/labels.csv \
  --label-mode csv \
  --output data/sequence/iscx_vpn_sequence.npz
```

### Useful options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-packets` | 32 | Truncate/pad each flow to this many packets |
| `--min-packets` | 1 | Skip flows with fewer packets |
| `--classes` | BROWSING,CHAT,… | Comma-separated valid labels |

## Train MTC-lite on sequence NPZ

`train_mtc_lite.py` is a **separate** training entry point for MTC-lite. It does not use `training/main.py` or the ARFF tabular pipeline.

Example smoke test (tiny synthetic dataset):

```bash
python training/sequence/train_mtc_lite.py \
  --data data/sequence/iscx_vpn_sequence.npz \
  --artifact-name mtc_lite_smoke \
  --epochs 5
```

Outputs:

- `models/<artifact-name>_sequence.pt` — PyTorch `state_dict`
- `models/<artifact-name>_sequence_results.json` — metrics and smoke-test flag

**Important:**

- Datasets with very few flows are **smoke tests only**; metrics are not meaningful benchmarks.
- A real comparison requires enough flows across all target classes.
- **XGBoost macro F1 0.8865** remains the deployment baseline to beat.
- ONNX export is not implemented yet for sequence models.

## Related files

- `build_sequence_dataset.py` — PCAP → NPZ builder
- `train_mtc_lite.py` — MTC-lite sequence training entry point
- `../models/mtc_lite.py` — PyTorch model expecting input `[batch, max_packets, num_features]`
- `../../docs/mtc_lite_research_plan.md` — research plan and evaluation criteria
- `../../configs/mtc_lite.yaml` — future sequence training config (not wired to ARFF `main.py` yet)
