import shlex
import shutil
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv312" / "bin" / "python"
PYTHON = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
WATCHED_FILES = (
    "main.py",
    "hybrid_engine.py",
    "sensor_simulator.py",
    "index.html",
    "static/camera.html",
)


def main():
    nodemon = shutil.which("nodemon")
    if nodemon is None:
        print(
            "Error: nodemon was not found. Install Node.js and nodemon, "
            "then run this script again.",
            file=sys.stderr,
        )
        return 1

    app_command = shlex.join(
        [
            str(PYTHON),
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
    )
    command = [nodemon]

    for watched_file in WATCHED_FILES:
        command.extend(["--watch", watched_file])

    command.extend(
        [
            "--ext",
            "py,html",
            "--exec",
            app_command,
        ]
    )

    print("Starting Campus Twin at http://localhost:8000")
    print(f"Using Python: {PYTHON}")
    print("Press Ctrl+C to stop the server.")

    try:
        return subprocess.run(command, cwd=BASE_DIR, check=False).returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
