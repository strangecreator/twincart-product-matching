from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    def from_cfg(cfg) -> ProjectPaths:
        # cfg.paths.* comes from configs/paths/default.yaml
        project_root = Path(cfg.paths.project_root).resolve()
        data_dir = Path(cfg.paths.data_dir).resolve()
        models_dir = Path(cfg.paths.models_dir).resolve()
        plots_dir = Path(cfg.paths.plots_dir).resolve()

        artifacts_dir = Path(cfg.paths.artifacts_dir).resolve()
        embeddings_dir = Path(cfg.paths.embeddings_dir).resolve()
        indices_dir = Path(cfg.paths.indices_dir).resolve()
        submissions_dir = Path(cfg.paths.submissions_dir).resolve()

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
        for p in [
            self.data_dir,
            self.models_dir,
            self.plots_dir,
            self.artifacts_dir,
            self.embeddings_dir,
            self.indices_dir,
            self.submissions_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)
