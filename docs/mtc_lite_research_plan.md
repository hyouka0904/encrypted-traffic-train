# MTC-lite Research Plan

## Current baseline

The best deployment model remains **XGBoost** (`configs/xgb.yaml`):

| Metric | Value |
|--------|------:|
| accuracy | 0.9120 |
| macro F1 | 0.8865 |
| ONNX size | 3964.9 KB |

LightGBM is close in accuracy but worse on macro F1 and produces a larger ONNX file.

## Why tabular DL did not win

We tested deep learning models (MLP, CNN1D, TabResNet) on the current **23-feature Scenario B-ARFF** tabular dataset. None beat XGBoost:

| Model | macro F1 | Notes |
|-------|---------:|-------|
| XGBoost | **0.8865** | Selected for deployment |
| LightGBM | 0.8807 | Close but worse |
| MLP | 0.5938 | Best DL on tabular features |
| CNN1D | 0.4260 | Paper-inspired, tabular input |
| TabResNet | 0.3613 | Experimental |

The ARFF features are aggregated flow statistics (packet counts, byte totals, durations, etc.). Tree models exploit these tabular patterns very well. Applying CNN or Transformer architectures to a flat 23-dimensional vector does not use the representation those papers assume.

## What the papers actually use

Models such as **Deep Packet**, **MTC (Multi-Task Classification)**, and **ET-BERT** outperform traditional flow-statistic classifiers mainly because they consume **richer packet-level or sequence-level representations**:

- Raw or partially raw packet bytes
- Per-packet length, direction, and timing
- Ordered sequences of packets within a flow
- In ET-BERT's case, tokenized byte sequences at datagram level

Beating XGBoost on encrypted traffic classification therefore requires a **new input pipeline**, not another tabular DL variant on the same ARFF features.

## Next step: raw packet sequence pipeline

The next meaningful research direction is **not** another tabular DL model. It is a **raw packet / packet-sequence preprocessing branch** that produces ordered flow sequences, then trains a dual-branch sequence model.

### MTC-lite target architecture

Inspired by MTC (Transformer + 1D-CNN), simplified for a first experiment:

**Input:** first K packets per flow, each packet represented by:

- packet length
- direction (client → server / server → client)
- inter-arrival time (IAT)
- optionally first N payload bytes (future extension)

**Branch A — 1D-CNN:** captures local sequential patterns (bursts, short n-grams of packet sizes/timing).

**Branch B — small Transformer encoder:** captures longer-range dependencies across the packet sequence.

**Fusion:** concatenate CNN and Transformer pooled embeddings.

**Output (v1):** single-task traffic category classification:

`BROWSING / CHAT / FT / MAIL / P2P / STREAMING / VOIP`

**Future (v2):** multi-task learning:

1. VPN vs non-VPN
2. traffic category
3. application

## Deployment policy

**Keep XGBoost as the deployment baseline** until MTC-lite (or a successor sequence model) beats **macro F1 0.8865** on a fair, held-out evaluation with the same class set.

Do not replace `configs/xgb.yaml` or overwrite `models/xgb.onnx` based on incomplete sequence experiments.

## Evaluation metrics

All sequence experiments should report the same metrics used for tree-model comparison:

| Metric | Purpose |
|--------|---------|
| **accuracy** | Overall correctness |
| **macro F1** | Primary comparison vs XGBoost (0.8865) |
| **ONNX size** | Deployability on resource-constrained AP |
| **inference latency** | Target: acceptable on Raspberry Pi AP (measure after ONNX export) |
| training time | Research cost tracking |

Use `output.name` in configs so experiments save separate artifacts (e.g. `models/mtc_lite_v1_results.json`) without overwriting the XGBoost baseline files.

## Risks and open problems

1. **Raw PCAP preprocessing** — ISCX-VPN provides PCAPs; flow segmentation and label alignment are non-trivial and not implemented yet.
2. **Label consistency** — sequence labels must match the seven traffic categories used in Scenario B-ARFF evaluation.
3. **Compute cost** — sequence models may train slower and use more memory than XGBoost.
4. **Model size** — dual-branch CNN + Transformer may exceed the ~4 MB XGBoost ONNX unless carefully sized.
5. **ONNX export** — `training/main.py` currently expects 2D tabular input; sequence models need sequence-aware dummy inputs and a dedicated training entry point.
6. **Raspberry Pi inference** — latency and memory must be verified on target hardware before any deployment switch.

## Repository scaffold (this branch)

| Path | Status |
|------|--------|
| `docs/mtc_lite_research_plan.md` | This document |
| `training/sequence/README.md` | Proposed NPZ dataset format |
| `training/sequence/build_sequence_dataset.py` | Placeholder (not implemented) |
| `training/models/mtc_lite.py` | PyTorch model definition only |
| `configs/mtc_lite.yaml` | Future config (not wired to ARFF `main.py`) |

No MTC-lite results are claimed until the sequence dataset exists and a training loop is implemented.
