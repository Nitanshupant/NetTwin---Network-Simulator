"""NetTwin launcher:  python main.py

1. Checks the Python version and required packages.
2. Creates/initialises the SQLite database (data/nettwin.db).
3. Starts the Streamlit dashboard in a child process.

Loop safety: Streamlit runs ``dashboard.py``, never this file, so it cannot
re-launch itself. As a second guard, a marker environment variable is set for
the child and ``main.py`` refuses to run if it ever finds that marker.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DASHBOARD = ROOT / "dashboard.py"
LAUNCH_MARKER = "NETTWIN_LAUNCHED"
REQUIRED_PACKAGES = {  # import name -> pip name
    "streamlit": "streamlit",
    "plotly": "plotly",
    "networkx": "networkx",
    "pandas": "pandas",
}
MIN_PYTHON = (3, 10)


def check_python() -> list[str]:
    if sys.version_info < MIN_PYTHON:
        return [f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required "
                f"(you have {sys.version_info.major}.{sys.version_info.minor})."]
    return []


def missing_packages() -> list[str]:
    return [pip for module, pip in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]


def database_path() -> Path:
    return Path(os.environ.get("NETTWIN_DB", ROOT / "data" / "nettwin.db"))


def initialise_database() -> Path:
    """Create the database and tables; raises DatabaseError with a clear message."""
    sys.path.insert(0, str(ROOT))
    from database import Database

    path = database_path()
    Database(path).initialize()
    return path


def find_free_port(preferred: int, attempts: int = 20) -> int:
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:  # nothing is listening
                return port
    raise RuntimeError(f"No free port found between {preferred} and {preferred + attempts - 1}.")


def build_command(port: int, open_browser: bool) -> list[str]:
    return [
        sys.executable, "-m", "streamlit", "run", str(DASHBOARD),
        "--server.port", str(port),
        "--server.headless", "false" if open_browser else "true",
        "--browser.gatherUsageStats", "false",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the NetTwin dashboard.")
    parser.add_argument("--port", type=int, default=8501, help="preferred port (default 8501)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    parser.add_argument("--check", action="store_true",
                        help="check dependencies and initialise the database, then exit")
    args = parser.parse_args(argv)

    if os.environ.get(LAUNCH_MARKER):
        print("NetTwin: main.py was started from inside the dashboard process; refusing to launch again.")
        return 2
    problems = check_python()
    if problems:
        print("NetTwin cannot start:\n  " + "\n  ".join(problems))
        return 1
    missing = missing_packages()
    if missing:
        print("NetTwin cannot start: missing packages: " + ", ".join(missing))
        print("Install them with:\n  python -m pip install -r requirements.txt")
        return 1
    if not DASHBOARD.exists():
        print(f"NetTwin cannot start: {DASHBOARD.name} is missing from {ROOT}.")
        return 1
    try:
        path = initialise_database()
    except Exception as exc:  # report any startup failure clearly, then stop
        print(f"NetTwin cannot start: the database could not be initialised ({exc}).")
        print("Check that the data/ folder is writable and the database file is not corrupt.")
        return 1
    print(f"Database ready: {path}")
    if args.check:
        print("All checks passed.")
        return 0
    try:
        port = find_free_port(args.port)
    except RuntimeError as exc:
        print(f"NetTwin cannot start: {exc}")
        return 1
    if port != args.port:
        print(f"Port {args.port} is busy; using {port} instead.")
    print(f"Starting NetTwin at http://localhost:{port}  (press Ctrl+C to stop)")
    env = dict(os.environ, **{LAUNCH_MARKER: "1"})
    try:
        return subprocess.run(build_command(port, not args.no_browser), cwd=ROOT, env=env).returncode
    except KeyboardInterrupt:
        print("\nNetTwin stopped.")
        return 0
    except OSError as exc:
        print(f"NetTwin cannot start Streamlit: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
