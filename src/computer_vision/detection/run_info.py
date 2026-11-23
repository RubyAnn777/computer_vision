"""Record what a training run used, so every result can be traced back.

`collect_run_info` gathers the git commit, whether the working tree had
uncommitted changes, the SHA-256 of the input files and any extra fields the
caller passes (settings, device, library versions, timings). The training
script writes it as ``run_info.json`` next to the Ultralytics results.
"""

import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        64-character hex digest.
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git_state(repo: Path) -> dict[str, str | bool]:
    """Return the current commit and whether the working tree is dirty.

    Parameters
    ----------
    repo : Path
        A folder inside the git repository.

    Returns
    -------
    dict[str, str | bool]
        ``{"commit": <hash>, "dirty": <bool>}``, or ``{"commit": "unknown",
        "dirty": True}`` when git is not available, so a run without git
        is never mistaken for a clean one.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}
    return {"commit": commit, "dirty": bool(status.strip())}


def collect_run_info(
    repo: Path, inputs: Mapping[str, Path], extra: Mapping[str, object]
) -> dict[str, object]:
    """Combine git state, input hashes and caller fields into one record.

    Parameters
    ----------
    repo : Path
        A folder inside the git repository.
    inputs : Mapping[str, Path]
        Named input files to fingerprint, e.g. the split file and weights.
    extra : Mapping[str, object]
        Further JSON-serialisable fields (settings, versions, timings).

    Returns
    -------
    dict[str, object]
        ``{"git": ..., "inputs_sha256": {...}, **extra}``.
    """
    return {
        "git": git_state(repo),
        "inputs_sha256": {name: sha256_of(p) for name, p in inputs.items()},
        **extra,
    }
