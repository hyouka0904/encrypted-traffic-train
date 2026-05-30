# Packet sequence dataset format (proposed)

This folder will hold preprocessing and loading code for **raw packet sequence** experiments (MTC-lite and successors).

**Status: not implemented.** No sequence dataset has been generated yet. The current training pipeline still uses tabular ARFF features via `training/preprocess.py` → `data/processed/`.

## Proposed NPZ format

Sequence datasets will be stored as NumPy compressed archives (`.npz`).

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `X` | `float32` | `[num_flows, max_packets, num_features]` | Per-flow packet sequences (padded/truncated to `max_packets`) |
| `y` | string array | `[num_flows]` | Traffic category labels |
| `feature_names` | list of str (optional) | — | Names for each feature dimension, e.g. `["packet_len", "direction", "iat"]` |

### Feature dimensions (initial proposal)

Each packet in a flow is described by:

- **packet_len** — length of the packet in bytes
- **direction** — `+1` or `-1` (or `0`/`1`) for client→server vs server→client
- **iat** — inter-arrival time since the previous packet in the flow (seconds)

Future versions may append normalized payload byte columns (`byte_0` … `byte_{N-1}`).

### Example output path

```
data/sequence/iscx_vpn_sequence.npz
```

This path is referenced by `configs/mtc_lite.yaml` but the file does not exist until `build_sequence_dataset.py` is implemented and run against ISCX-VPN PCAPs.

## Related files

- `build_sequence_dataset.py` — placeholder PCAP → NPZ script (exit code 1 until implemented)
- `../models/mtc_lite.py` — PyTorch model expecting input `[batch, max_packets, num_features]`
- `../../docs/mtc_lite_research_plan.md` — full research plan and evaluation criteria
