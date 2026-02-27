"""Compilation matrix dataclasses and cross-product expansion."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import yaml


@dataclass
class ISAConfig:
    name: str
    chost: str  # Gentoo target triplet, e.g. "aarch64-unknown-linux-gnu"

    @property
    def is_native(self) -> bool:
        return self.name == "x86_64"

    @property
    def emerge_cmd(self) -> str:
        if self.is_native:
            return "emerge"
        return f"{self.chost}-emerge"


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
        if self.compiler_family == "gcc":
            if self.isa.is_native:
                return f"gcc-{self.compiler_version}"
            return f"{self.isa.chost}-gcc-{self.compiler_version}"
        else:
            # Clang is inherently a cross-compiler
            cc = f"clang-{self.compiler_version}"
            if not self.isa.is_native:
                cc += f" --target={self.isa.chost} --sysroot=/usr/{self.isa.chost}"
            return cc

    @property
    def cxx(self) -> str:
        if self.compiler_family == "gcc":
            if self.isa.is_native:
                return f"g++-{self.compiler_version}"
            return f"{self.isa.chost}-g++-{self.compiler_version}"
        else:
            cxx = f"clang++-{self.compiler_version}"
            if not self.isa.is_native:
                cxx += f" --target={self.isa.chost} --sysroot=/usr/{self.isa.chost}"
            return cxx

    @property
    def cflags(self) -> str:
        return f"{self.opt_level} -g"

    @property
    def env_tag(self) -> str:
        """Tag used for /etc/portage/env/ file naming."""
        return self.config_tag


def load_matrix(config_path: Path) -> list[CompilationConfig]:
    """Load compiler matrix from YAML and expand the full cross-product."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    isas = [
        ISAConfig(name=isa["name"], chost=isa["chost"])
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
