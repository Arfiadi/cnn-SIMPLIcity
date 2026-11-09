"""Run integrity and distribution audit on SIMPLIcity dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simplicity_cnn.data.audit import audit_dataset
from simplicity_cnn.data.dataset import CLASS_NAMES
from simplicity_cnn.utils.env import get_default_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SIMPLIcity dataset integrity and splits.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset root folder (defaults to auto-detected path).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save JSON audit report.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else get_default_data_dir()
    print(f"Auditing dataset at: {data_dir}")

    report = audit_dataset(data_dir)

    print("\n" + "=" * 60)
    print("SIMPLIcity Dataset Audit Report")
    print("=" * 60)
    print(f"Root Directory: {report['root_dir']}")
    print(
        f"Overall Health: {'[PASSED] Healthy' if report['is_healthy'] else '[FAILED] Corrupted files detected'}"
    )
    print(f"Color Spaces:   {dict(report['channel_stats'])}")

    for split, details in report["splits"].items():
        print(f"\n--- Split: {split.upper()} ---")
        if "error" in details:
            print(f"  Status: ERROR - {details['error']}")
            continue
        print(f"  Total Images:    {details['total_images']}")
        print(f"  Dimension Min:   {details['min_dimension']}")
        print(f"  Dimension Max:   {details['max_dimension']}")
        print("  Class Distribution:")
        for cls_idx, count in details["class_distribution"].items():
            name = (
                CLASS_NAMES[cls_idx]
                if isinstance(cls_idx, int) and cls_idx < len(CLASS_NAMES)
                else str(cls_idx)
            )
            print(f"    Class {cls_idx:2d} ({name:<10}): {count} images")

    if report["corrupted_files"]:
        print("\n[WARNING] Corrupted Files:")
        for cf in report["corrupted_files"]:
            print(f"  - {cf}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved full report to: {out_p}")


if __name__ == "__main__":
    main()
