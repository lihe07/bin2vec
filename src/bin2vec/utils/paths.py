"""Canonical path helpers and config tag utilities."""

from __future__ import annotations

from pathlib import Path

from bin2vec.config.matrix import CompilationConfig
from bin2vec.config.packages import PackageConfig


def data_root(data_dir: str = "data") -> Path:
    return Path(data_dir).resolve()


def sources_dir(data_dir: str = "data") -> Path:
    return data_root(data_dir) / "sources"


def builds_dir(
    data_dir: str,
    pkg: PackageConfig,
    config: CompilationConfig,
) -> Path:
    return data_root(data_dir) / "builds" / pkg.name / pkg.version / config.config_tag


def extracted_dir(data_dir: str, pkg: PackageConfig) -> Path:
    return data_root(data_dir) / "extracted" / pkg.name / pkg.version


def extracted_parquet(
    data_dir: str,
    pkg: PackageConfig,
    config: CompilationConfig,
) -> Path:
    return extracted_dir(data_dir, pkg) / f"{config.config_tag}.parquet"


def dataset_dir(data_dir: str = "data") -> Path:
    return data_root(data_dir) / "dataset"


def parse_config_tag(tag: str) -> dict[str, str]:
    """Parse a config tag like 'x86_64_gcc-14_O2' into components."""
    parts = tag.split("_")
    isa = parts[0]
    compiler = parts[1]  # e.g. "gcc-14"
    opt = parts[2]  # e.g. "O2"
    family, version = compiler.split("-")
    return {
        "isa": isa,
        "compiler_family": family,
        "compiler_version": version,
        "opt_level": f"-{opt}",
    }
