# Model Results Summary

## Selected deployment model

The selected deployment model is the original XGBoost baseline:

| Model | Accuracy | Macro F1 | ONNX size |
|---|---:|---:|---:|
| `xgb.yaml` | 0.9120 | 0.8865 | 3964.9 KB |

This model is selected because it achieves the best macro F1 and accuracy among all tested models while keeping a reasonable ONNX size for deployment.

## Full comparison

| Model / Config | Accuracy | Macro F1 | ONNX size | Decision |
|---|---:|---:|---:|---|
| `xgb.yaml` | 0.9120 | 0.8865 | 3964.9 KB | Selected deployment model |
| `lgb.yaml` | 0.9078 | 0.8807 | 4378.6 KB | Close, but worse and larger |
| `xgb_n200_d6_lr010` | 0.9078 | 0.8814 | 2828.6 KB | Smaller, but rejected due to F1 drop |
| `mlp.yaml` | 0.6844 | 0.5938 | 196.4 KB | Best DL baseline, but weaker than tree models |
| `cnn1d.yaml` | 0.6114 | 0.4260 | 272.4 KB | Paper-inspired experiment, not selected |
| `tabresnet` local test | 0.4152 | 0.3613 | 564.6 KB | Removed after poor result |

## XGBoost tuning conclusion

The smaller XGBoost candidate `xgb_n200_d6_lr010` reduced ONNX size from 3964.9 KB to 2828.6 KB, about a 28.7% reduction.

However, macro F1 dropped from 0.8865 to 0.8814. Because the F1 drop was outside the accepted window, the smaller candidate was rejected for deployment.

## Deep learning conclusion

MLP, CNN1D, and TabResNet did not beat XGBoost on the current Scenario B-ARFF feature dataset.

The likely reason is that the current dataset is based on tabular flow features, while stronger encrypted traffic classification papers usually use richer packet-level or sequence-level representations.

## Next research direction

XGBoost remains the deployment baseline.

The next research direction is MTC-lite, which requires a new raw packet sequence pipeline:

- Parse PCAP files
- Group packets into flows
- Extract packet length, direction, and inter-arrival time
- Save sequence data as NPZ
- Train MTC-lite on packet sequences
- Compare against XGBoost macro F1 0.8865
