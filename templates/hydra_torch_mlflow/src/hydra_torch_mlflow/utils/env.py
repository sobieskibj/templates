from pathlib import Path
import os


def get_outputs_dir() -> Path:
    path = os.environ.get("OUTPUTS_DIR", None)
    if path is None:
        raise ValueError("Could not get the OUTPUTS_DIR env variable")
    path = Path(path)
    if not path.exists():
        raise ValueError("OUTPUTS_DIR directory does not exist.")
    return path
