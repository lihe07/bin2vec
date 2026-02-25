"""Walk build outputs, parse ELF files, filter functions, write Parquet."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from bin2vec.config.matrix import CompilationConfig
from bin2vec.config.packages import PackageConfig
from bin2vec.extract.elf_parser import ElfParser
from bin2vec.extract.function_filter import should_keep
from bin2vec.utils.logging import get_logger
from bin2vec.utils.paths import builds_dir, extracted_parquet

log = get_logger("extractor")

SCHEMA = pa.schema([
    ("function_name", pa.string()),
    ("source_file", pa.string()),
    ("package_name", pa.string()),
    ("package_version", pa.string()),
    ("compiler_family", pa.string()),
    ("compiler_version", pa.int32()),
    ("opt_level", pa.string()),
    ("isa", pa.string()),
    ("config_tag", pa.string()),
    ("address", pa.uint64()),
    ("size", pa.uint32()),
    ("raw_bytes", pa.binary()),
])


def _find_elf_files(directory: Path) -> list[Path]:
    """Find all ELF files in a directory tree."""
    elf_files = []
    if not directory.exists():
        return elf_files

    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        # Check ELF magic bytes
        try:
            with open(path, "rb") as f:
                magic = f.read(4)
                if magic == b"\x7fELF":
                    elf_files.append(path)
        except (PermissionError, OSError):
            continue
    return elf_files


def extract_config(
    pkg: PackageConfig,
    config: CompilationConfig,
    data_dir: str,
) -> int:
    """Extract functions from all ELF files for a single (pkg, config).

    Returns the number of functions extracted.
    """
    build_path = builds_dir(data_dir, pkg, config)
    out_path = extracted_parquet(data_dir, pkg, config)

    # Resume: skip if parquet already exists
    if out_path.exists():
        log.debug("Skipping existing extraction: %s", out_path)
        table = pq.read_table(out_path)
        return len(table)

    elf_files = _find_elf_files(build_path)
    if not elf_files:
        log.debug("No ELF files found in %s", build_path)
        return 0

    rows: list[dict] = []

    for elf_path in elf_files:
        parser = ElfParser(elf_path)
        functions = parser.extract_functions()

        for func in functions:
            if not should_keep(func):
                continue

            rows.append({
                "function_name": func.name,
                "source_file": func.source_file,
                "package_name": pkg.name,
                "package_version": pkg.version,
                "compiler_family": config.compiler_family,
                "compiler_version": config.compiler_version,
                "opt_level": config.opt_level,
                "isa": config.isa.name,
                "config_tag": config.config_tag,
                "address": func.address,
                "size": func.size,
                "raw_bytes": func.raw_bytes,
            })

    if not rows:
        log.debug("No functions extracted from %s/%s", pkg.name, config.config_tag)
        return 0

    # Write Parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({col: [r[col] for r in rows] for col in SCHEMA.names}, schema=SCHEMA)
    pq.write_table(table, out_path)

    log.info("Extracted %d functions: %s/%s", len(rows), pkg.name, config.config_tag)
    return len(rows)


def extract_all(
    packages: list[PackageConfig],
    configs: list[CompilationConfig],
    data_dir: str,
) -> int:
    """Extract functions from all (pkg, config) combinations.

    Returns total number of functions extracted.
    """
    total = 0
    tasks = [(pkg, config) for pkg in packages for config in configs]

    with tqdm(total=len(tasks), desc="Extracting", unit="config") as pbar:
        for pkg, config in tasks:
            count = extract_config(pkg, config, data_dir)
            total += count
            pbar.update(1)

    log.info("Total functions extracted: %d", total)
    return total
