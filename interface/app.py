"""
CVRP-IA interface entry point.

Run with either of::

    python -m interface                  (canonical, from project root)
    python interface/app.py              (direct-script form)

CRITICAL: matplotlib's backend is set to ``TkAgg`` *before* importing
anything from ``src.*`` to avoid other modules binding ``Agg``.
"""
from __future__ import annotations

# 0. Make the project root importable so that ``interface`` (this package)
#    and ``src`` can both be found whether we are launched as
#    ``python -m interface.app`` (parent on sys.path) or as
#    ``python interface/app.py`` (parent NOT on sys.path).
import os
import sys as _sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT and _PARENT not in _sys.path:
    _sys.path.insert(0, _PARENT)

# 1. Set matplotlib backend FIRST — must precede any pyplot-using import.
import matplotlib

matplotlib.use("TkAgg")

# 2. Standard library / third-party imports.
import argparse
import sys
import tkinter as tk

# 3. Local imports (may transitively import pyplot).
from interface.main_window import MainWindow


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m interface",
        description=(
            "Interfaccia grafica (Tkinter) per CVRP Immunological Algorithm.\n"
            "Permette di configurare i parametri dell'algoritmo e di "
            "osservare l'evoluzione delle run in tempo reale."
        ),
    )
    parser.add_argument(
        "--instances-dir",
        type=str,
        default="instances",
        help="Directory delle istanze .vrp (default: ./instances)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # Override the default instances dir from ParamPanel-relative path
    # through MainWindow by patching the constant before construction.
    MainWindow.INSTANCES_DIR = args.instances_dir  # type: ignore[attr-defined]

    root = tk.Tk()
    try:
        MainWindow(root)
    except Exception as exc:  # noqa: BLE001
        import traceback
        print("Failed to launch interface:", exc, file=sys.stderr)
        traceback.print_exc()
        return 1

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
