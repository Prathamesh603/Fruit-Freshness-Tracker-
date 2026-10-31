"""Start only the project-owned headless emulator; print its PID for later cleanup."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
sdk = TOOLS / "android-sdk"
environment = os.environ.copy()
environment.update({"ANDROID_HOME": str(sdk), "ANDROID_SDK_ROOT": str(sdk),
                    "ANDROID_AVD_HOME": str(TOOLS / "avd"), "ANDROID_USER_HOME": str(TOOLS / "android-user"),
                    "ANDROID_EMULATOR_HOME": str(TOOLS / "android-user"), "ANDROID_ADB_SERVER_PORT": "5038"})
(TOOLS / "android-user").mkdir(exist_ok=True)
with (TOOLS / "emulator.log").open("w", encoding="utf-8") as output:
    process = subprocess.Popen([str(sdk / "emulator/emulator.exe"), "-avd", "FruitLensTest", "-no-window", "-no-audio",
        "-no-boot-anim", "-no-snapshot", "-gpu", "swiftshader", "-port", "5556", "-memory", "1536"],
        env=environment, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW)
(TOOLS / "emulator-process.json").write_text(json.dumps({"pid": process.pid, "avd": "FruitLensTest", "port": 5556}), encoding="utf-8")
print(f"Started project test emulator PID {process.pid}; log: {TOOLS / 'emulator.log'}")
