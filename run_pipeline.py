"""Runs the three pipeline stages once: data engineering → model engineering → deployment."""
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "raw" / "diabetes.csv"
COMPOSE_FILE = ROOT / "code" / "deployment" / "docker-compose.yml"


def _compose_command() -> list:
    """Prefer the `docker compose` plugin, fall back to legacy `docker-compose`."""
    for candidate in (["docker", "compose"], ["docker-compose"]):
        try:
            subprocess.run(candidate + ["version"], capture_output=True, check=True)
            return candidate
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    raise RuntimeError("Docker Compose not found – is Docker installed and running?")


def _run_stage(name: str, command: list) -> None:
    print("\n" + "=" * 72)
    print(f"  {name}")
    print("=" * 72)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Stage '{name}' failed (exit code {result.returncode}).")


def main() -> None:
    started = time.perf_counter()
    print(f"Pipeline started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not RAW_DATA.exists():
        _run_stage("STAGE 0 – Download raw data", [sys.executable, "-m", "code.datasets.download_data"])

    _run_stage("STAGE 1 – Data engineering", [sys.executable, "-m", "code.datasets.prepare_data"])
    _run_stage("STAGE 2 – Model engineering", [sys.executable, "-m", "code.models.train_model"])
    _run_stage(
        "STAGE 3 – Deployment (build & start API + app containers)",
        _compose_command() + ["-f", str(COMPOSE_FILE), "up", "-d", "--build"],
    )

    print(f"\nPipeline finished in {time.perf_counter() - started:.1f} s")
    print("  • App: http://localhost:8501")
    print("  • API: http://localhost:8000/docs")


if __name__ == "__main__":
    main()