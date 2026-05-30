"""Build packet-sequence NPZ datasets from raw PCAPs.

PLACEHOLDER — raw PCAP-to-flow sequence extraction is not implemented yet.
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build packet-sequence NPZ dataset from PCAP files (not implemented).",
    )
    parser.add_argument(
        "--pcap-dir",
        required=True,
        help="Directory containing ISCX-VPN PCAP files",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to label mapping or metadata CSV/JSON aligned with PCAPs",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output NPZ path, e.g. data/sequence/iscx_vpn_sequence.npz",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=32,
        help="Maximum packets per flow to retain (default: 32)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        "ERROR: Raw PCAP-to-flow sequence extraction is not implemented yet.\n"
        "\n"
        "This script is a placeholder for the future MTC-lite preprocessing pipeline.\n"
        "Planned steps (TODO):\n"
        "  1. Parse PCAPs under --pcap-dir\n"
        "  2. Group packets into flows (5-tuple + timing)\n"
        "  3. Extract per-packet: length, direction, inter-arrival time\n"
        "  4. Align flow labels with --labels\n"
        "  5. Pad/truncate to --max-packets and save NPZ to --output\n"
        "\n"
        f"Received: pcap-dir={args.pcap_dir!r}, labels={args.labels!r}, "
        f"output={args.output!r}, max-packets={args.max_packets}\n"
        "\n"
        "See training/sequence/README.md and docs/mtc_lite_research_plan.md.",
        file=sys.stderr,
    )

    # TODO: parse PCAPs (e.g. scapy or dpkt)
    # TODO: group packets into flows
    # TODO: extract packet length, direction, inter-arrival time
    # TODO: align labels with Scenario B-ARFF class names
    # TODO: save NPZ with keys X, y, feature_names

    sys.exit(1)


if __name__ == "__main__":
    main()
