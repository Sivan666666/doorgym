"""Backward-compatible wrapper for the renamed Door policy play entrypoint."""

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from play_door_policy import main  # noqa: E402


if __name__ == "__main__":
    main()
