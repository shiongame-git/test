"""Run a script against this Windows/Python 3.13 source checkout."""

from pathlib import Path
import runpy
import sys


def main():
    if sys.platform != "win32" or sys.version_info[:2] != (3, 13):
        raise SystemExit("Packing3D requires Windows and Python 3.13.")
    root = Path(__file__).resolve().parent
    script = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / "demo" / "gui.pyw"
    sys.path[:0] = [str(script.parent), str(root / "src")]
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
