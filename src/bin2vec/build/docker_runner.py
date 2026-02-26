"""Docker container lifecycle management for builds."""

from __future__ import annotations

from pathlib import Path

import docker
from docker.errors import ImageNotFound

from bin2vec.utils.logging import get_logger

log = get_logger("docker")


class DockerRunner:
    """Manages Docker container lifecycle for compilation builds."""

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

    def ensure_image(self, image_name: str, dockerfile_dir: Path, isa: str) -> None:
        """Build Docker image if it doesn't already exist."""
        try:
            self.client.images.get(image_name)
            log.debug("Image %s already exists", image_name)
        except ImageNotFound:
            log.info("Building Docker image %s from %s", image_name, dockerfile_dir)
            self.client.images.build(
                path=str(dockerfile_dir),
                dockerfile=f"Dockerfile.{isa}",
                tag=image_name,
                rm=True,
            )
            log.info("Image %s built successfully", image_name)

    def run_build(
        self,
        image_name: str,
        build_script: str,
        sources_path: Path,
        output_path: Path,
    ) -> tuple[bool, str]:
        """Run a build script inside a Docker container.

        Returns (success, log_output).
        """
        output_path.mkdir(parents=True, exist_ok=True)

        container = self.client.containers.run(
            image=image_name,
            command=["bash", "-c", build_script],
            volumes={
                str(sources_path): {"bind": "/workspace/sources", "mode": "ro"},
                str(output_path): {"bind": "/workspace/output", "mode": "rw"},
            },
            tmpfs={"/workspace/build": ""},
            detach=True,
            mem_limit="4g",
            labels={self.LABEL: "1"},
        )

        try:
            result = container.wait(timeout=1800)  # 30 min timeout
            logs = container.logs().decode("utf-8", errors="replace")
            exit_code = result["StatusCode"]

            if exit_code != 0:
                log.warning("Build failed (exit %d): %s", exit_code, logs[-500:])
                return False, logs

            return True, logs
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

    def close(self) -> None:
        self.client.close()
