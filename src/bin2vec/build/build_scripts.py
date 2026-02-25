"""Generate bash build scripts per (package build-system × compilation config)."""

from __future__ import annotations

from bin2vec.config.matrix import CompilationConfig
from bin2vec.config.packages import PackageConfig


def generate_build_script(
    pkg: PackageConfig,
    config: CompilationConfig,
    source_dir: str,
    output_dir: str,
) -> str:
    """Generate a bash build script for a package + compilation config.

    The script is meant to run inside a Docker container with sources
    mounted at /workspace/sources and output at /workspace/output.
    """
    cc = config.cc
    cxx = config.cxx
    cflags = config.cflags

    lines = [
        "#!/bin/bash",
        "set -e",
        f"cd /workspace/sources/{source_dir}",
        "",
        f'export CC="{cc}"',
        f'export CXX="{cxx}"',
        f'export CFLAGS="{cflags}"',
        f'export CXXFLAGS="{cflags}"',
        "",
    ]

    if pkg.build_system == "autotools":
        configure_args = " ".join(pkg.build_options.get("configure_args", []))
        host_flag = ""
        if config.isa.cross_prefix:
            # Strip trailing dash for --host
            host = config.isa.cross_prefix.rstrip("-")
            host_flag = f"--host={host}"

        lines += [
            "mkdir -p /workspace/build && cd /workspace/build",
            f'/workspace/sources/{source_dir}/configure {host_flag} {configure_args} '
            f'CC="$CC" CFLAGS="$CFLAGS" CXX="$CXX" CXXFLAGS="$CXXFLAGS" '
            f'--prefix=/workspace/output',
            "make -j$(nproc)",
            "make install DESTDIR=/workspace/output || make install prefix=/workspace/output || true",
        ]

    elif pkg.build_system == "cmake":
        cmake_args = " ".join(pkg.build_options.get("cmake_args", []))
        toolchain = ""
        if config.isa.cross_prefix:
            # For cross-compilation with CMake, set system name
            system_map = {
                "aarch64-linux-gnu-": "Linux",
                "mipsel-linux-gnu-": "Linux",
            }
            system = system_map.get(config.isa.cross_prefix, "Linux")
            processor_map = {
                "aarch64-linux-gnu-": "aarch64",
                "mipsel-linux-gnu-": "mipsel",
            }
            processor = processor_map.get(config.isa.cross_prefix, "")
            toolchain = (
                f'-DCMAKE_SYSTEM_NAME={system} '
                f'-DCMAKE_SYSTEM_PROCESSOR={processor} '
                f'-DCMAKE_C_COMPILER={cc} '
                f'-DCMAKE_CXX_COMPILER={cxx} '
                f'-DCMAKE_C_FLAGS="{cflags}" '
                f'-DCMAKE_CXX_FLAGS="{cflags}"'
            )
        else:
            toolchain = (
                f'-DCMAKE_C_COMPILER={cc} '
                f'-DCMAKE_CXX_COMPILER={cxx} '
                f'-DCMAKE_C_FLAGS="{cflags}" '
                f'-DCMAKE_CXX_FLAGS="{cflags}"'
            )

        lines += [
            "mkdir -p /workspace/build && cd /workspace/build",
            f"cmake /workspace/sources/{source_dir} "
            f"-DCMAKE_INSTALL_PREFIX=/workspace/output "
            f"{toolchain} {cmake_args}",
            "make -j$(nproc)",
            "make install",
        ]

    elif pkg.build_system == "custom":
        configure_cmd = pkg.build_options.get("configure_cmd", "")
        build_cmd = pkg.build_options.get("build_cmd", "")

        # Template variable substitution
        subs = {
            "{CC}": cc,
            "{CXX}": cxx,
            "{CFLAGS}": cflags,
            "{CXXFLAGS}": cflags,
            "{CROSS_PREFIX}": config.cross_prefix,
            "{OPENSSL_TARGET}": config.isa.openssl_target,
        }
        for k, v in subs.items():
            configure_cmd = configure_cmd.replace(k, v)
            build_cmd = build_cmd.replace(k, v)

        if configure_cmd:
            lines.append(configure_cmd)
        lines.append(build_cmd)
        lines.append("cp -r . /workspace/output/ 2>/dev/null || true")

    lines.append("")
    return "\n".join(lines)
