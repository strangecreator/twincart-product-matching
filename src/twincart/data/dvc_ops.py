from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Optional


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def dvc_pull(paths: Iterable[Path], remote: Optional[str] = None) -> None:
    args = ["dvc", "pull"]
    if remote:
        args += ["-r", remote]
    args += [str(p) for p in paths]
    _run(args)


def ensure_exists_or_pull(required: Iterable[Path], remote: Optional[str] = None) -> None:
    missing = [p for p in required if not p.exists()]

    if not missing:
        return

    # Pull minimal required targets (parents may be dirs)
    dvc_pull(missing, remote=remote)
