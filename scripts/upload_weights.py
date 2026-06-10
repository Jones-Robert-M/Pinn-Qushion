#!/usr/bin/env python
"""Upload production weights to a HuggingFace model repository.

Usage:
    python scripts/upload_weights.py --repo your-hf-username/pinn-qushion-weights

The repo must already exist on HuggingFace (create it at huggingface.co/new).
Set HF_TOKEN env var or pass --token.
"""

import argparse
import os
from pathlib import Path

PRODUCTION_WEIGHTS = [
    "infinite_well.eqx",
    "harmonic.eqx",
    "finite_well.eqx",
    "double_well.eqx",
    "gaussian_well.eqx",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. username/pinn-qushion-weights")  # noqa: E501
    parser.add_argument("--weights-dir", default="weights", help="Local weights directory")
    parser.add_argument("--token", default=None, help="HF token (falls back to HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Provide --token or set HF_TOKEN environment variable")

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    weights_dir = Path(args.weights_dir)
    uploaded, skipped = [], []

    for fname in PRODUCTION_WEIGHTS:
        path = weights_dir / fname
        if not path.exists():
            skipped.append(fname)
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=fname,
            repo_id=args.repo,
            repo_type="model",
        )
        uploaded.append(fname)
        print(f"Uploaded: {fname}")

    print(f"\nDone. Uploaded {len(uploaded)}, skipped {len(skipped)}")
    if skipped:
        print(f"Skipped (not found): {skipped}")
    print(f"\nSet HF_WEIGHTS_REPO={args.repo} in your GitHub repo secrets.")


if __name__ == "__main__":
    main()
