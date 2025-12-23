from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_dir: Path
    models_dir: Path
    plots_dir: Path
    artifacts_dir: Path
    embeddings_dir: Path
    indices_dir: Path
    submissions_dir: Path

    @staticmethod
    def from_cfg(config) -> ProjectPaths:
        project_root = Path(config.paths.project_root).resolve()
        data_dir = Path(config.paths.data_dir).resolve()
        models_dir = Path(config.paths.models_dir).resolve()
        plots_dir = Path(config.paths.plots_dir).resolve()

        artifacts_dir = Path(config.paths.artifacts_dir).resolve()
        embeddings_dir = Path(config.paths.embeddings_dir).resolve()
        indices_dir = Path(config.paths.indices_dir).resolve()
        submissions_dir = Path(config.paths.submissions_dir).resolve()

        return ProjectPaths(
            project_root=project_root,
            data_dir=data_dir,
            models_dir=models_dir,
            plots_dir=plots_dir,
            artifacts_dir=artifacts_dir,
            embeddings_dir=embeddings_dir,
            indices_dir=indices_dir,
            submissions_dir=submissions_dir,
        )

    def ensure_dirs(self) -> None:
        for path in [
            self.data_dir,
            self.models_dir,
            self.plots_dir,
            self.artifacts_dir,
            self.embeddings_dir,
            self.indices_dir,
            self.submissions_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
