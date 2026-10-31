"""Install app/test APKs on the isolated emulator and run actual model inference."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ADB = ROOT / ".tools/android-sdk/platform-tools/adb.exe"
REPORTS = ROOT / "verification"
REPORTS.mkdir(exist_ok=True)


def adb(*arguments, binary=False, check=True):
    result = subprocess.run([str(ADB), "-P", "5038", "-s", "emulator-5556", *arguments],
                            capture_output=True, text=not binary, check=check)
    if not binary:
        print(result.stdout, flush=True)
    return result.stdout


def main():
    adb("logcat", "-c")
    adb("install", "-r", str(ROOT / "app/build/outputs/apk/debug/app-x86_64-debug.apk"))
    adb("install", "-r", str(ROOT / "app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"))
    result = adb("shell", "am", "instrument", "-w", "-r", "com.fruitlens.classic.test/androidx.test.runner.AndroidJUnitRunner")
    (REPORTS / "instrumentation.txt").write_text(result, encoding="utf-8")
    logs = adb("logcat", "-d", "-s", "FruitLensTest:I", "AndroidRuntime:E", "*:S")
    (REPORTS / "inference-logcat.txt").write_text(logs, encoding="utf-8")
    if "OK (4 tests)" not in result:
        raise RuntimeError("Inference smoke tests did not all pass. See verification/instrumentation.txt.")
    screenshot = adb("exec-out", "run-as", "com.fruitlens.classic", "cat", "files/smoke-result.png", binary=True)
    (REPORTS / "smoke-result.png").write_bytes(screenshot)
    print(f"Smoke tests passed. Screenshot: {REPORTS / 'smoke-result.png'}")


if __name__ == "__main__":
    main()
