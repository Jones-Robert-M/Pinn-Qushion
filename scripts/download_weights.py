#!/usr/bin/env python
"""Download pre-trained PINN weights from HuggingFace.

Usage:
    python scripts/download_weights.py

Weights are saved to weights/ and are ready for use with scripts/train_all.py
(warm-start) or the Streamlit app.
"""

import argparse
from pathlib import Path

PRODUCTION_WEIGHTS = [
    "infinite_well.eqx",
    "harmonic.eqx",
    "finite_well.eqx",
    "double_well.eqx",
    "gaussian_well.eqx",
]

DEFAULT_REPO = "JonesRobM/pinn-qushion-weights"


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained PINN weights")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"HuggingFace model repo (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--output-dir",
        default="weights",
        help="Local directory to save weights (default: weights/)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token for private repos (not needed for public repos)",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Downloading weights from {args.repo} -> {output_dir}/")
    print()

    downloaded, failed = [], []

    for fname in PRODUCTION_WEIGHTS:
        try:
            local_path = hf_hub_download(
                repo_id=args.repo,
                filename=fname,
                repo_type="model",
                local_dir=str(output_dir),
                token=args.token,
            )
            src = Path(local_path)
            dst = output_dir / fname
            if src != dst:
                import shutil
                shutil.copy2(src, dst)
            size_kb = dst.stat().st_size // 1024
            print(f"  {fname} ({size_kb} KB)")
            downloaded.append(fname)
        except Exception as e:
            print(f"  {fname} — FAILED: {e}")
            failed.append(fname)

    print()
    print(f"Downloaded {len(downloaded)}/{len(PRODUCTION_WEIGHTS)} weight files.")
    if failed:
        print(f"Failed: {failed}")
    else:
        print("All weights ready. Run `streamlit run app.py` to launch the app.")


if __name__ == "__main__":
    main()
