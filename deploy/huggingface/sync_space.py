"""Mirror this repository into a Hugging Face **Docker** Space.

Invoked by ``.github/workflows/deploy-hf-space.yml``. Reads two environment
variables:

* ``HF_TOKEN``  - a Hugging Face token with write scope.
* ``HF_SPACE``  - the target repo id, e.g. ``your-username/MultiDiseaseAI``.

It creates the Space if missing, uploads the whole working tree (LFS-tracked
model bundles included) minus local cruft, and finally overwrites the Space's
``README.md`` with ``deploy/huggingface/README.md`` so the Space card / SDK
metadata is correct.
"""

from __future__ import annotations

import os
import pathlib
import sys

from huggingface_hub import HfApi

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SPACE_CARD = REPO_ROOT / "deploy" / "huggingface" / "README.md"

IGNORE = [
    ".git/*",
    ".git/**",
    ".venv/*",
    ".venv/**",
    "**/__pycache__/**",
    "*.pyc",
    ".pytest_cache/**",
    "notebooks/**",
    "docs/screenshots/**",
    "reports/generated/*.pdf",
    "reports/prediction_history.db",
    ".tabpfn_models/**",
    "README.md",  # replaced with the Space card below
]


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    space = os.environ.get("HF_SPACE")
    if not token or not space:
        print("HF_TOKEN and HF_SPACE must both be set", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)

    api.upload_folder(
        folder_path=str(REPO_ROOT),
        repo_id=space,
        repo_type="space",
        ignore_patterns=IGNORE,
        commit_message="Sync from GitHub",
    )
    api.upload_file(
        path_or_fileobj=str(SPACE_CARD),
        path_in_repo="README.md",
        repo_id=space,
        repo_type="space",
        commit_message="Update Space card",
    )
    print(f"Synced {REPO_ROOT} -> https://huggingface.co/spaces/{space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
