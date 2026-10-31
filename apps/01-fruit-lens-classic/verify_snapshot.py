"""Check that the recorded Classic app, models, source, and documentation are intact."""
from pathlib import Path
import hashlib
import json
import sys


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "SNAPSHOT_MANIFEST.json").read_text(encoding="utf-8"))
    failed = []
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if root not in path.parents:
            failed.append(f"Invalid manifest path: {relative}")
        elif not path.is_file():
            failed.append(f"Missing: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failed.append(f"Changed: {relative}")
    if failed:
        print("\n".join(failed))
        return 1
    print(f"Classic snapshot intact: {len(manifest['files'])} recorded files match SHA-256.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
