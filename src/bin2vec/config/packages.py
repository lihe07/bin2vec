"""Gentoo package discovery and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class PackageConfig:
    name: str
    category: str
    version: str
    atom: str  # e.g. "dev-libs/openssl-3.3.1"

    @property
    def cp(self) -> str:
        """Category/package atom without version (e.g. 'dev-libs/openssl')."""
        return f"{self.category}/{self.name}"


def load_categories(config_path: Path) -> list[str]:
    """Load category list from YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data["categories"]
