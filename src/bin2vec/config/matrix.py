"""Compilation matrix dataclasses and cross-product expansion."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import yaml


@dataclass
class ISAConfig:
    name: str
    docker_image: str
    cross_prefix: str
    openssl_target: str


@dataclass
class CompilationConfig:
    compiler_family: str  # "gcc" or "clang"
    compiler_version: int
    opt_level: str  # e.g. "-O2"
    isa: ISAConfig

    @property
    def config_tag(self) -> str:
        return f"{self.isa.name}_{self.compiler_family}-{self.compiler_version}_{self.opt_level.replace('-', '')}"

    @property
    def cc(self) -> str:
        prefix = self.isa.cross_prefix
        if self.compiler_family == "gcc":
            return f"{prefix}gcc-{self.compiler_version}"
        else:
            return f"clang-{self.compiler_version}"

    @property
    def cxx(self) -> str:
        prefix = self.isa.cross_prefix
        if self.compiler_family == "gcc":
            return f"{prefix}g++-{self.compiler_version}"
        else:
            return f"clang++-{self.compiler_version}"

    @property
    def cflags(self) -> str:
        flags = f"{self.opt_level} -g"
        if self.compiler_family == "clang" and self.isa.cross_prefix:
            target_map = {
                "aarch64-linux-gnu-": "aarch64-linux-gnu",
                "mipsel-linux-gnu-": "mipsel-linux-gnu",
            }
            target = target_map.get(self.isa.cross_prefix, "")
            if target:
                flags += f" --target={target} --sysroot=/usr/{target}"
        return flags

    @property
    def cross_prefix(self) -> str:
        return self.isa.cross_prefix


def load_matrix(config_path: Path) -> list[CompilationConfig]:
    """Load compiler matrix from YAML and expand the full cross-product."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    isas = [
        ISAConfig(
            name=isa["name"],
            docker_image=isa["docker_image"],
            cross_prefix=isa["cross_prefix"],
            openssl_target=isa["openssl_target"],
        )
        for isa in data["isas"]
    ]

    compilers = []
    for comp in data["compilers"]:
        for ver in comp["versions"]:
            compilers.append((comp["family"], ver))

    opt_levels = data["optimization_levels"]

    configs = []
    for (family, version), opt, isa in product(compilers, opt_levels, isas):
        configs.append(
            CompilationConfig(
                compiler_family=family,
                compiler_version=version,
                opt_level=opt,
                isa=isa,
            )
        )
    return configs


def filter_matrix(
    configs: list[CompilationConfig],
    *,
    isas: list[str] | None = None,
    compilers: list[str] | None = None,
    opt_levels: list[str] | None = None,
) -> list[CompilationConfig]:
    """Filter a list of CompilationConfigs by ISA, compiler family, or opt level."""
    result = configs
    if isas:
        result = [c for c in result if c.isa.name in isas]
    if compilers:
        result = [c for c in result if c.compiler_family in compilers]
    if opt_levels:
        result = [c for c in result if c.opt_level in opt_levels]
    return result
