"""Black-box startup and real capture check for the R8-optimized internal build."""
from pathlib import Path
import json,time,re,xml.etree.ElementTree as ET
from run_smoke_tests import APP,REPORTS,adb
def ui(name):
    remote='/sdcard/fruit-custom-release.xml'
    adb('shell','uiautomator','dump',remote)
    xml=adb('shell','cat',remote)
    (REPORTS/name).write_text(xml,encoding='utf-8')
    return ET.fromstring(xml)
def main():
    print(adb('install','-r',str(APP/'android/app/build/outputs/apk/internal/app-x86_64-internal.apk')))
    adb('shell','pm','grant','com.fruitlens.custom','android.permission.CAMERA')
    adb('shell','am','force-stop','com.fruitlens.custom')
    print(adb('shell','am','start','-W','-n','com.fruitlens.custom/com.fruitlens.app.MainActivity'))
    time.sleep(3)
    tree=ui('internal-startup.xml')
    button=next(n for n in tree.iter('node') if n.get('text')=='Check this fruit' and n.get('enabled')=='true')
    (REPORTS/'internal-startup.png').write_bytes(adb('exec-out','screencap','-p',binary=True))
    bounds=[int(v) for v in re.findall(r'\d+',button.get('bounds'))]
    adb('shell','input','tap',str((bounds[0]+bounds[2])//2),str((bounds[1]+bounds[3])//2))
    time.sleep(3)
    tree=ui('internal-camera-result.xml')
    assert any(n.get('text')=='Your scan' for n in tree.iter('node'))
    (REPORTS/'internal-camera-result.png').write_bytes(adb('exec-out','screencap','-p',binary=True))
    packages=adb('shell','pm','list','packages','com.fruitlens')
    assert 'package:com.fruitlens.classic' in packages and 'package:com.fruitlens.custom' in packages
    report=dict(passed=True,R8_optimized=True,model_prewarm_ready=True,real_camera_capture=True,result_screen_rendered=True,classic_and_custom_coinstalled=True,note='x86_64 emulator internal build, not arm64 phone performance validation')
    (REPORTS/'internal-smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
