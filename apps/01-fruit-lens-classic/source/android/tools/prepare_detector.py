"""Fetch pinned official NCNN runtime and NanoDet-Plus-m detection weights."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
ASSETS = ROOT / "app/src/main/assets/detector"
VERSION = "20261020"
MODEL_TAG = "v1.0.0-alpha-1"


def fetch(url, target):
    if not target.exists():
        print(f"Downloading {target.name}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, target.with_suffix(".part").open("wb") as output:
            shutil.copyfileobj(response, output)
        target.with_suffix(".part").replace(target)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def main():
    TOOLS.mkdir(exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    runtime_url = f"https://github.com/Tencent/ncnn/releases/download/{VERSION}/ncnn-{VERSION}-android.zip"
    runtime = TOOLS / f"ncnn-{VERSION}-android.zip"
    runtime_hash = fetch(runtime_url, runtime)
    destination = TOOLS / "ncnn"
    if not destination.exists():
        with zipfile.ZipFile(runtime) as archive:
            for name in archive.namelist():
                path = (destination / name).resolve()
                if destination.resolve() not in path.parents:
                    raise ValueError("Unexpected archive path")
            archive.extractall(destination)
    if all((ASSETS / name).exists() for name in ("nanodet.param", "nanodet.bin", "provenance.json")):
        recorded = json.loads((ASSETS / "provenance.json").read_text(encoding="utf-8"))
        for name in ("nanodet.param", "nanodet.bin"):
            actual = hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()
            if actual != recorded["files"][name]:
                raise ValueError(f"Preserved Classic model changed: {name}")
        print("NCNN runtime prepared; preserved Classic detector assets left unchanged.", flush=True)
        return
    model_url = f"https://github.com/RangiLyu/nanodet/releases/download/{MODEL_TAG}/nanodet-plus-m_416_ncnn.zip"
    model_zip = TOOLS / "nanodet-plus-m_416_ncnn.zip"
    model_zip_hash = fetch(model_url, model_zip)
    with zipfile.ZipFile(model_zip) as archive:
        names = [name for name in archive.namelist() if name.endswith((".param", ".bin"))]
        for suffix in (".param", ".bin"):
            selected = [name for name in names if name.endswith(suffix)]
            if len(selected) != 1:
                raise ValueError("Expected one param/bin pair")
            (ASSETS / ("nanodet" + suffix)).write_bytes(archive.read(selected[0]))
    license_path = ASSETS / "NANODET_LICENSE.txt"
    license_url = "https://raw.githubusercontent.com/RangiLyu/nanodet/be9b4a9001d7f9b6fc89c2df31ae8d428e35b4f0/LICENSE"
    fetch(license_url, license_path)
    ncnn_license = ASSETS / "NCNN_LICENSE.txt"
    fetch(f"https://raw.githubusercontent.com/Tencent/ncnn/{VERSION}/LICENSE.txt", ncnn_license)
    provenance = {"model": "NanoDet-Plus-m", "inputSize": 416, "source": model_url,
                  "archiveSha256": model_zip_hash, "runtime": VERSION, "runtimeSource": runtime_url,
                  "runtimeArchiveSha256": runtime_hash,
                  "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ASSETS.iterdir() if p.is_file()}}
    (ASSETS / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print((ASSETS / "nanodet.param").read_text(encoding="utf-8")[-3500:], flush=True)
    print(f"Detector ready at {ASSETS}", flush=True)


if __name__ == "__main__":
    main()
