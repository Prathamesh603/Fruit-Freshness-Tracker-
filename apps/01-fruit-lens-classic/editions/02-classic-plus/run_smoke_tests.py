"""Exercise the Custom app's actual Android models on the project-owned emulator."""
from pathlib import Path
import subprocess,json,time
APP=Path(__file__).resolve().parent
ROOT=APP.parents[3]
ADB=ROOT/'android/.tools/android-sdk/platform-tools/adb.exe'
REPORTS=APP/'verification'
def adb(*args,binary=False):
    p=subprocess.run([str(ADB),'-P','5038','-s','emulator-5556',*args],capture_output=True,text=not binary,check=True)
    return p.stdout
def main():
    REPORTS.mkdir(exist_ok=True)
    adb('wait-for-device')
    deadline=time.monotonic()+90
    while adb('shell','getprop','sys.boot_completed').strip()!='1' and time.monotonic()<deadline:
        time.sleep(1)
    assert adb('shell','getprop','sys.boot_completed').strip()=='1'
    adb('logcat','-c')
    for apk in ['debug/app-x86_64-debug.apk','androidTest/debug/app-debug-androidTest.apk']:
        print(adb('install','-r',str(APP/'android/app/build/outputs/apk'/apk)),flush=True)
    result=adb('shell','am','instrument','-w','-r','com.fruitlens.classic.plus.test/androidx.test.runner.AndroidJUnitRunner')
    (REPORTS/'instrumentation.txt').write_text(result)
    logs=adb('logcat','-d','-s','FruitLensTest:I','AndroidRuntime:E','*:S')
    (REPORTS/'inference-logcat.txt').write_text(logs)
    assert 'OK (5 tests)' in result,result
    screenshot=adb('exec-out','run-as','com.fruitlens.classic.plus','cat','files/smoke-result.png',binary=True)
    (REPORTS/'smoke-result.png').write_bytes(screenshot)
    (REPORTS/'multi-fruit-result.png').write_bytes(adb('exec-out','run-as','com.fruitlens.classic.plus','cat','files/multi-fruit-result.png',binary=True))
    packages=adb('shell','pm','list','packages','com.fruitlens')
    (REPORTS/'coinstallation.json').write_text(json.dumps(dict(packages=packages.splitlines(),classic_plus_installed='package:com.fruitlens.classic.plus' in packages,classic_installed='package:com.fruitlens.classic' in packages,note='Distinct IDs, no replacement of Classic'),indent=2))
    print(result,flush=True)
if __name__=='__main__':main()
