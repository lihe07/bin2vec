"""Docker container lifecycle management for Gentoo builds."""

from __future__ import annotations

from pathlib import Path

import docker
from docker.errors import ImageNotFound

from bin2vec.utils.logging import get_logger

log = get_logger("docker")

IMAGE_NAME = "bin2vec-gentoo"

# Path inside every build container where Portage stores binary packages.
# This must match PKGDIR in the Dockerfile's make.conf.
CONTAINER_PKGDIR = "/var/cache/binpkgs"


class DockerRunner:
    """Manages a long-lived Gentoo container for compilation builds."""

    LABEL = "bin2vec.build"

    def __init__(self) -> None:
        self.client = docker.from_env()
        self._cleanup_stale_containers()

    def _cleanup_stale_containers(self) -> None:
        """Remove any stopped containers from previous runs."""
        stale = self.client.containers.list(
            all=True,
            filters={"label": self.LABEL, "status": "exited"},
        )
        if stale:
            log.info("Cleaning up %d stale build containers", len(stale))
            for c in stale:
                try:
                    c.remove(force=True)
                except Exception:
                    pass

    def ensure_image(self, dockerfile_dir: Path) -> None:
        """Build the Gentoo Docker image if it doesn't already exist."""
        try:
            self.client.images.get(IMAGE_NAME)
            log.debug("Image %s already exists", IMAGE_NAME)
        except ImageNotFound:
            log.info("Building Docker image %s from %s", IMAGE_NAME, dockerfile_dir)
            self.client.images.build(
                path=str(dockerfile_dir),
                dockerfile="Dockerfile",
                tag=IMAGE_NAME,
                rm=True,
            )
            log.info("Image %s built successfully", IMAGE_NAME)

    def start_container(
        self,
        output_path: Path,
        binpkg_dir: Path | None = None,
    ) -> str:
        """Start a long-lived Gentoo container. Returns the container ID.

        Args:
            output_path: Host path to mount as ``/workspace/output`` (build
                         artefacts – ELF binaries).
            binpkg_dir:  Host path to mount as ``/var/cache/binpkgs`` (binary
                         package cache).  When supplied, Portage reuses
                         already-compiled ``.gpkg`` files from here and writes
                         new ones back, making subsequent runs much faster.
                         If *None*, no cache volume is mounted (packages are
                         always compiled from source).
        """
        output_path.mkdir(parents=True, exist_ok=True)

        volumes: dict[str, dict] = {
            str(output_path): {"bind": "/workspace/output", "mode": "rw"},
        }

        if binpkg_dir is not None:
            binpkg_dir.mkdir(parents=True, exist_ok=True)
            volumes[str(binpkg_dir)] = {
                "bind": CONTAINER_PKGDIR,
                "mode": "rw",
            }
            log.info("Binpkg cache mounted: %s → %s", binpkg_dir, CONTAINER_PKGDIR)
        else:
            log.debug(
                "No binpkg cache mounted; all packages will be compiled from source"
            )

        container = self.client.containers.run(
            image=IMAGE_NAME,
            command=["sleep", "infinity"],
            volumes=volumes,
            detach=True,
            mem_limit="8g",
            labels={self.LABEL: "1"},
            privileged=False,
        )

        log.info("Started build container: %s", container.short_id)
        assert container.id is not None, "Docker SDK returned a container with no ID"
        return container.id

    def exec_in_container(
        self,
        container_id: str,
        command: list[str],
        timeout: int = 600,
    ) -> tuple[int, str]:
        """Execute a command inside the running container.

        Returns (exit_code, output).
        """
        container = self.client.containers.get(container_id)
        exec_result = container.exec_run(
            command,
            demux=False,
        )
        output = (
            exec_result.output.decode("utf-8", errors="replace")
            if exec_result.output
            else ""
        )
        return exec_result.exit_code, output

    def stop_container(self, container_id: str) -> None:
        """Stop and remove the build container."""
        try:
            container = self.client.containers.get(container_id)
            container.stop(timeout=10)
            container.remove(force=True)
            log.info("Stopped container: %s", container_id[:12])
        except Exception as e:
            log.warning("Failed to stop container %s: %s", container_id[:12], e)

    def copy_from_container(
        self,
        container_id: str,
        src_path: str,
        dst_path: Path,
    ) -> bool:
        """Copy files from the container to the host."""
        import tarfile
        import io

        container = self.client.containers.get(container_id)
        try:
            bits, _ = container.get_archive(src_path)
            dst_path.mkdir(parents=True, exist_ok=True)
            stream = io.BytesIO()
            for chunk in bits:
                stream.write(chunk)
            stream.seek(0)
            with tarfile.open(fileobj=stream) as tar:
                tar.extractall(path=dst_path)
            return True
        except Exception as e:
            log.warning("Failed to copy %s from container: %s", src_path, e)
            return False

    def close(self) -> None:
        self.client.close()
