from __future__ import annotations

import pathlib
import subprocess
import typing as tp


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def dvc_pull(paths: tp.Iterable[pathlib.Path], remote: tp.Optional[str] = None) -> None:
    args = ["dvc", "pull"]

    if remote:
        args += ["-r", remote]

    args += [str(path) for path in paths]

    _run(args)


def _find_repo_root(start: pathlib.Path) -> pathlib.Path:
    current = start.resolve()

    for parent in [current, *current.parents]:
        if (parent / ".dvc").exists():
            return parent

    raise RuntimeError("Cannot find .dvc directory. (are you in the repo?)")


def ensure_exists_or_pull(paths: tp.Iterable[pathlib.Path], remote: str | None = None) -> None:
    paths = [pathlib.Path(path) for path in paths]
    missing = [path for path in paths if not path.exists()]

    if not missing:
        return

    repo_root = _find_repo_root(pathlib.Path(__file__).resolve())

    try:
        from dvc.repo import Repo
    except Exception as exc:
        raise RuntimeError("dvc is required to pull missing files. Install with `uv sync --all-extras`.") from exc

    with Repo(str(repo_root)) as repo:
        repo.pull(targets=[str(path) for path in missing], remote=remote)

    still_missing = [path for path in paths if not path.exists()]

    if still_missing:
        raise FileNotFoundError(f"DVC pull did not materialize: {still_missing}.")
