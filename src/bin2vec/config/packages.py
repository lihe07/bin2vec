"""Package configuration dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SourceConfig:
    type: str  # "tarball", "git", or "apt"
    url: str | None = None
    branch: str | None = None
    package: str | None = None  # apt source package name


@dataclass
class PackageConfig:
    name: str
    version: str
    source: SourceConfig
    build_system: str  # "autotools", "cmake", "custom"
    build_options: dict = field(default_factory=dict)

    @property
    def source_dir_name(self) -> str:
        return f"{self.name}-{self.version}"


def load_packages(config_path: Path) -> list[PackageConfig]:
    """Load package configs from YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    packages = []
    for pkg in data["packages"]:
        source = SourceConfig(
            type=pkg["source"]["type"],
            url=pkg["source"].get("url"),
            branch=pkg["source"].get("branch"),
            package=pkg["source"].get("package"),
        )
        packages.append(
            PackageConfig(
                name=pkg["name"],
                version=pkg["version"],
                source=source,
                build_system=pkg["build_system"],
                build_options=pkg.get("build_options", {}),
            )
        )
    return packages
