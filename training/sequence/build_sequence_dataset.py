"""Build packet-sequence NPZ datasets from raw PCAPs.

First real implementation for the MTC-lite research pipeline.
Extracts per-flow packet sequences (length, direction, IAT) from IPv4 TCP/UDP
traffic and saves a compressed NPZ for future sequence-model training.

Does not integrate with training/main.py (ARFF tabular path).
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from scapy.all import IP, TCP, UDP, PcapReader
except ImportError as exc:
    raise SystemExit(
        "scapy is required. Install with: pip install scapy"
    ) from exc

FEATURE_NAMES = ["packet_len", "direction", "iat"]
DEFAULT_CLASSES = ["BROWSING", "CHAT", "FT", "MAIL", "P2P", "STREAMING", "VOIP"]
PCAP_SUFFIXES = {".pcap", ".pcapng"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build packet-sequence NPZ dataset from PCAP files.",
    )
    parser.add_argument(
        "--pcap-dir",
        required=True,
        help="Directory containing .pcap / .pcapng files (searched recursively)",
    )
    parser.add_argument(
        "--output",
        default="data/sequence/iscx_vpn_sequence.npz",
        help="Output NPZ path (default: data/sequence/iscx_vpn_sequence.npz)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional CSV file mapping PCAP basename to label (required for --label-mode csv)",
    )
    parser.add_argument(
        "--label-mode",
        choices=["filename", "csv"],
        default="filename",
        help="How to assign labels to flows (default: filename)",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=32,
        help="Maximum packets per flow (default: 32)",
    )
    parser.add_argument(
        "--min-packets",
        type=int,
        default=1,
        help="Minimum packets required to keep a flow (default: 1)",
    )
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated known class names",
    )
    return parser.parse_args()


def parse_classes(raw: str) -> list[str]:
    classes = [part.strip().upper() for part in raw.split(",") if part.strip()]
    if not classes:
        raise ValueError("--classes must contain at least one class name")
    return classes


def find_pcap_files(pcap_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(pcap_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in PCAP_SUFFIXES:
            files.append(path)
    return files


def load_csv_labels(labels_path: Path, classes: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV labels file is empty: {labels_path}")

        normalized_fields = {name.strip().lower(): name for name in reader.fieldnames}
        if "file" not in normalized_fields or "label" not in normalized_fields:
            raise ValueError(
                f"CSV labels must include 'file' and 'label' columns: {labels_path}"
            )

        file_col = normalized_fields["file"]
        label_col = normalized_fields["label"]

        for row in reader:
            filename = row[file_col].strip()
            label = row[label_col].strip().upper()
            if not filename or not label:
                continue
            if label not in classes:
                warnings.warn(
                    f"Skipping CSV row with unknown label {label!r} for file {filename!r}"
                )
                continue
            mapping[filename] = label
    return mapping


def infer_label_from_path(path: Path, classes: list[str]) -> str | None:
    """Infer class from PCAP filename or parent directory names (case-insensitive)."""
    search_parts = [path.name, path.stem, *[parent.name for parent in path.parents]]
    classes_by_len = sorted(classes, key=len, reverse=True)

    for part in search_parts:
        part_lower = part.lower()
        for cls in classes_by_len:
            if cls.lower() in part_lower:
                return cls
    return None


def resolve_pcap_label(
    pcap_path: Path,
    label_mode: str,
    classes: list[str],
    csv_labels: dict[str, str],
) -> str | None:
    if label_mode == "csv":
        return csv_labels.get(pcap_path.name)

    return infer_label_from_path(pcap_path, classes)


def canonical_flow_key_and_direction(
    proto: int,
    src_ip: str,
    src_port: int,
    dst_ip: str,
    dst_port: int,
) -> tuple[tuple[int, str, int, str, int], float]:
    """Return bidirectional flow key and packet direction (+1 / -1)."""
    src = (src_ip, src_port)
    dst = (dst_ip, dst_port)

    if src <= dst:
        a_ip, a_port, b_ip, b_port = src_ip, src_port, dst_ip, dst_port
    else:
        a_ip, a_port, b_ip, b_port = dst_ip, dst_port, src_ip, src_port

    key = (proto, a_ip, a_port, b_ip, b_port)
    direction = 1.0 if (src_ip, src_port) == (a_ip, a_port) else -1.0
    return key, direction


def extract_packet_record(packet) -> tuple[tuple[int, str, int, str, int], float, float, float] | None:
    """Extract flow key, direction, packet length, and timestamp from one Scapy packet."""
    if IP not in packet:
        return None

    ip_layer = packet[IP]
    if TCP in packet:
        proto = 6
        l4 = packet[TCP]
    elif UDP in packet:
        proto = 17
        l4 = packet[UDP]
    else:
        return None

    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    src_port = int(l4.sport)
    dst_port = int(l4.dport)

    flow_key, direction = canonical_flow_key_and_direction(
        proto, src_ip, src_port, dst_ip, dst_port
    )
    timestamp = float(packet.time)
    packet_len = float(len(packet))
    return flow_key, direction, packet_len, timestamp


def packets_to_flow_sequences(
    flow_packets: dict[tuple[int, str, int, str, int], list[tuple[float, float, float]]],
    max_packets: int,
    min_packets: int,
) -> list[np.ndarray]:
    """Convert raw per-flow packet records into padded [max_packets, 3] arrays."""
    sequences: list[np.ndarray] = []

    for records in flow_packets.values():
        if len(records) < min_packets:
            continue

        records.sort(key=lambda item: item[0])  # timestamp order

        seq = np.zeros((max_packets, 3), dtype=np.float32)
        prev_time: float | None = None

        for idx, (timestamp, packet_len, direction) in enumerate(records[:max_packets]):
            if prev_time is None:
                iat = 0.0
            else:
                iat = max(timestamp - prev_time, 0.0)

            seq[idx, 0] = packet_len
            seq[idx, 1] = direction
            seq[idx, 2] = float(iat)
            prev_time = timestamp

        sequences.append(seq)

    return sequences


def process_pcap(
    pcap_path: Path,
    max_packets: int,
    min_packets: int,
) -> list[np.ndarray]:
    """Read one PCAP and return flow sequence arrays."""
    flow_packets: dict[tuple[int, str, int, str, int], list[tuple[float, float, float]]] = (
        defaultdict(list)
    )

    try:
        with PcapReader(str(pcap_path)) as reader:
            for packet in reader:
                try:
                    extracted = extract_packet_record(packet)
                    if extracted is None:
                        continue
                    flow_key, direction, packet_len, timestamp = extracted
                    flow_packets[flow_key].append((timestamp, packet_len, direction))
                except Exception as exc:
                    warnings.warn(
                        f"{pcap_path}: skipping malformed packet ({exc})"
                    )
    except Exception as exc:
        warnings.warn(f"{pcap_path}: failed to read PCAP ({exc})")
        return []

    return packets_to_flow_sequences(flow_packets, max_packets, min_packets)


def build_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    pcap_dir = Path(args.pcap_dir)
    if not pcap_dir.is_dir():
        raise SystemExit(f"--pcap-dir does not exist or is not a directory: {pcap_dir}")

    classes = parse_classes(args.classes)
    class_set = set(classes)

    if args.label_mode == "csv":
        if not args.labels:
            raise SystemExit("--labels is required when --label-mode csv")
        labels_path = Path(args.labels)
        if not labels_path.is_file():
            raise SystemExit(f"Labels CSV not found: {labels_path}")
        csv_labels = load_csv_labels(labels_path, class_set)
    else:
        csv_labels = {}

    pcap_files = find_pcap_files(pcap_dir)
    if not pcap_files:
        raise SystemExit(f"No .pcap / .pcapng files found under {pcap_dir}")

    x_rows: list[np.ndarray] = []
    y_rows: list[str] = []

    pcaps_used = 0
    pcaps_skipped = 0

    for pcap_path in pcap_files:
        label = resolve_pcap_label(pcap_path, args.label_mode, classes, csv_labels)
        if label is None:
            pcaps_skipped += 1
            if args.label_mode == "csv":
                warnings.warn(
                    f"Skipping {pcap_path}: no label found in CSV for basename {pcap_path.name!r}"
                )
            else:
                warnings.warn(
                    f"Skipping {pcap_path}: could not infer label from filename or parent directories"
                )
            continue

        sequences = process_pcap(pcap_path, args.max_packets, args.min_packets)
        if not sequences:
            pcaps_skipped += 1
            warnings.warn(f"Skipping {pcap_path}: no usable flows extracted")
            continue

        x_rows.extend(sequences)
        y_rows.extend([label] * len(sequences))
        pcaps_used += 1

    if not x_rows:
        raise SystemExit("No flows were extracted. Check PCAP contents and label settings.")

    x = np.stack(x_rows, axis=0).astype(np.float32)
    y = np.array(y_rows, dtype=str)
    return x, y, len(pcap_files), pcaps_used, pcaps_skipped


def print_summary(
    pcap_found: int,
    pcaps_used: int,
    pcaps_skipped: int,
    output_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    classes: list[str],
) -> None:
    print("\n=== Sequence dataset summary ===")
    print(f"PCAP files found   : {pcap_found}")
    print(f"PCAP files used    : {pcaps_used}")
    print(f"PCAP files skipped : {pcaps_skipped}")
    print(f"Flows saved        : {x.shape[0]}")
    print(f"Output path        : {output_path}")
    print(f"X shape            : {tuple(x.shape)}")
    print("Class distribution :")
    counts = Counter(y.tolist())
    for cls in classes:
        if counts.get(cls, 0) > 0:
            print(f"  {cls:<12} {counts[cls]}")
    for cls, count in sorted(counts.items()):
        if cls not in classes:
            print(f"  {cls:<12} {count}")


def main() -> None:
    args = parse_args()
    classes = parse_classes(args.classes)

    x, y, pcap_found, pcaps_used, pcaps_skipped = build_dataset(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=x,
        y=y,
        feature_names=np.array(FEATURE_NAMES, dtype=str),
        max_packets=np.int32(args.max_packets),
        classes=np.array(classes, dtype=str),
    )

    print_summary(pcap_found, pcaps_used, pcaps_skipped, output_path, x, y, classes)
    print("\n[done] sequence dataset saved")


if __name__ == "__main__":
    main()
