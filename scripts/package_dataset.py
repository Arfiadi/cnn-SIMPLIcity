"""Package dataset folder into a portable zip file for quick upload to Colab or Kaggle."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# Ensure repository root src/ directory is available in sys.path
_repo_src = Path(__file__).resolve().parent.parent / "src"
if _repo_src.is_dir() and str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

from simplicity_cnn.utils.env import get_default_data_dir


def package_dataset(source_dir: Path | str, output_zip: Path | str) -> Path:
    """Create a zip archive of the dataset.

    Args:
        source_dir: Path to directory containing train, validation, and test folders.
        output_zip: Destination zip path.

    Returns:
        Path: Path to created zip file.
    """
    src = Path(source_dir).resolve()
    out = Path(output_zip).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    print(f"Packaging dataset from: {src}")
    file_count = 0

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in src.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rel_path = file_path.relative_to(src)
                zf.write(file_path, arcname=str(rel_path))
                file_count += 1

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Successfully packaged {file_count} images into: {out} ({size_mb:.2f} MB)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package SIMPLIcity dataset into zip for Colab/Kaggle."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset containing train/validation/test folders (defaults to auto-detected path).",
    )
    parser.add_argument(
        "--output-zip",
        type=str,
        default="simplicity_dataset.zip",
        help="Output zip filename/path.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else get_default_data_dir()
    package_dataset(data_dir, args.output_zip)


if __name__ == "__main__":
    main()
