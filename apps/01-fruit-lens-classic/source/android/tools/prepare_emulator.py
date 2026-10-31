"""Create an isolated Android 11 x86_64 emulator for native inference smoke tests."""
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
from bootstrap_android import ROOT, TOOLS, download, extract, get_text


def main():
    sdk = TOOLS / "android-sdk"
    image = sdk / "system-images/android-30/default/x86_64"
    if not (image / "system.img").exists():
        base = "https://dl.google.com/android/repository/sys-img/android/"
        repository = ET.fromstring(get_text(base + "sys-img2-1.xml"))
        packages = [p for p in repository if p.tag.endswith("remotePackage") and p.get("path") == "system-images;android-30;default;x86_64"]
        package = max(packages, key=lambda p: int(p.findtext("revision/major", "0")))
        complete = package.find("archives/archive/complete")
        archive = TOOLS / Path(complete.findtext("url")).name
        checksum = complete.find("checksum")
        download(base + complete.findtext("url"), archive, checksum.get("type", "sha1"), checksum.text.strip())
        stage = TOOLS / "extract-test-image"
        extract(archive, stage)
        root = next(stage.rglob("system.img")).parent
        image.parent.mkdir(parents=True, exist_ok=True)
        root.rename(image)
    avd = TOOLS / "avd/FruitLensTest.avd"
    avd.mkdir(parents=True, exist_ok=True)
    config = {
        "avd.ini.encoding": "UTF-8", "abi.type": "x86_64", "hw.cpu.arch": "x86_64",
        "hw.cpu.ncore": "2", "hw.ramSize": "1536", "hw.lcd.width": "432", "hw.lcd.height": "960",
        "hw.lcd.density": "180", "hw.gpu.enabled": "yes", "hw.gpu.mode": "swiftshader_indirect",
        "hw.keyboard": "yes", "hw.camera.back": "emulated", "hw.camera.front": "none",
        "image.sysdir.1": image.as_posix() + "/", "tag.id": "default", "tag.display": "Default",
        "disk.dataPartition.size": "1G", "showDeviceFrame": "no", "fastboot.forceColdBoot": "yes",
    }
    (avd / "config.ini").write_text("\n".join(f"{k}={v}" for k, v in config.items()) + "\n", encoding="utf-8")
    (avd.parent / "FruitLensTest.ini").write_text(f"avd.ini.encoding=UTF-8\npath={avd.as_posix()}\ntarget=android-30\n", encoding="utf-8")
    print(f"Test AVD ready: {avd}")


if __name__ == "__main__":
    main()
