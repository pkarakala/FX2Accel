"""Simple CLI entrypoint for FX2Accel."""

from __future__ import annotations

import argparse
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fx2accel")
    parser.add_argument("--version", action="store_true", help="Show version")
    args = parser.parse_args(argv)
    if args.version:
        from . import __version__

        print(f"fx2accel {__version__}")
        return 0
    print("FX2Accel placeholder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
