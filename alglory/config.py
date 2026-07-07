"""Application configuration and filesystem layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

UI_DIR = Path(__file__).parent / "ui"


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    db_path: Path
    ui_dir: Path

    @classmethod
    def default(cls) -> "AppConfig":
        home = Path(os.environ.get("ALGLORY_HOME", Path.home() / ".alglory"))
        home.mkdir(parents=True, exist_ok=True)
        (home / "bars").mkdir(exist_ok=True)
        (home / "exports").mkdir(exist_ok=True)
        return cls(data_dir=home, db_path=home / "vault.db", ui_dir=UI_DIR)
