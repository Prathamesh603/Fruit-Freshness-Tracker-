"""Download a checksum-verified, project-local Android/Gradle build toolchain."""
from pathlib import Path
import hashlib
import os
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
TOOLS.mkdir(parents=True, exist_ok=True)


def download(url, destination, algorithm, expected):
    if destination.exists() and hashlib.new(algorithm, destination.read_bytes()).hexdigest() == expected:
        return
    print(f"Downloading {destination.name}", flush=True)
    partial = destination.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    actual = hashlib.new(algorithm, partial.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {destination.name}")
    partial.replace(destination)


def extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            resolved = (destination / item.filename).resolve()
            if resolved != destination and destination not in resolved.parents:
                raise RuntimeError("Archive member escapes destination")
        package.extractall(destination)


def get_text(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def main():
    if os.name != "nt":
        raise SystemExit("This bootstrap currently targets Windows. Use Android Studio on other hosts.")
    base = "https://services.gradle.org/distributions/gradle-8.13-bin.zip"
    archive = TOOLS / "gradle-8.13-bin.zip"
    download(base, archive, "sha256", get_text(base + ".sha256").strip().split()[0])
    gradle = TOOLS / "gradle-8.13"
    if not (gradle / "bin/gradle.bat").exists():
        extract(archive, TOOLS)

    repository = "https://dl.google.com/android/repository/"
    document = ET.fromstring(get_text(repository + "repository2-1.xml"))
    sdk = TOOLS / "android-sdk"
    packages = [("platforms;android-36", sdk / "platforms/android-36"),
                ("build-tools;36.0.0", sdk / "build-tools/36.0.0"),
                ("platform-tools", sdk / "platform-tools")]
    if "--emulator" in sys.argv:
        packages += [("emulator", sdk / "emulator")]
    if "--native" in sys.argv:
        packages += [("ndk;28.2.13676358", sdk / "ndk/28.2.13676358"),
                     ("cmake;3.22.1", sdk / "cmake/3.22.1")]
    for package_path, output_path in packages:
        if (output_path / "source.properties").exists():
            properties = (output_path / "source.properties").read_text(encoding="utf-8")
            if package_path == "platforms;android-36" and "AndroidVersion.IsBaseSdk=false" in properties:
                # Preserve the extension package at its actual SDK path rather than misidentifying it.
                extension = next(line.split("=", 1)[1].strip() for line in properties.splitlines()
                                 if line.startswith("AndroidVersion.ExtensionLevel="))
                suffix = "android-36-ext" + extension
                output_path.rename(output_path.parent / suffix)
            else:
                continue
        matches = [p for p in document if p.tag.endswith("remotePackage") and p.get("path") == package_path]
        if not matches:
            raise RuntimeError(f"SDK package not in official repository: {package_path}")
        package = max(matches, key=lambda p: int(p.findtext("revision/major", "0")))
        archives = package.find("archives")
        selected = next(a for a in archives if a.findtext("host-os") in (None, "windows"))
        complete = selected.find("complete")
        url = complete.findtext("url")
        checksum = complete.find("checksum")
        algorithm = checksum.get("type", "sha1")
        zip_path = TOOLS / Path(url).name
        download(repository + url, zip_path, algorithm, checksum.text.strip())
        staging = TOOLS / ("extract-" + output_path.name.replace(".", "-"))
        staging.mkdir(exist_ok=True)
        extract(zip_path, staging)
        package_root = next(staging.rglob("source.properties")).parent
        output_path.parent.mkdir(parents=True, exist_ok=True)
        package_root.rename(output_path)

    (ROOT / "local.properties").write_text("sdk.dir=" + sdk.as_posix() + "\n", encoding="utf-8")
    print(f"Toolchain ready: {gradle / 'bin/gradle.bat'}", flush=True)
    print("SDK packages were downloaded from Google; their bundled license files remain in place.", flush=True)


if __name__ == "__main__":
    main()
