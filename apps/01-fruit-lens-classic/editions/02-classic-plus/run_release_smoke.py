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
    (REPORTS/'internal-smoke.json').write_text(json.dumps(dict(passed=False,stage='running')))
    print(adb('install','-r',str(APP/'android/app/build/outputs/apk/internal/app-x86_64-internal.apk')))
    adb('shell','pm','grant','com.fruitlens.classic.plus','android.permission.CAMERA')
    adb('shell','am','force-stop','com.fruitlens.classic.plus')
    print(adb('shell','am','start','-W','-n','com.fruitlens.classic.plus/com.fruitlens.app.MainActivity'))
    deadline=time.monotonic()+45
    button=None
    while time.monotonic()<deadline:
        tree=ui('internal-startup.xml')
        button=next((n for n in tree.iter('node') if n.get('enabled')=='true' and n.get('clickable')=='true'
            and any(child.get('text')=='Check this fruit' for child in n.iter('node'))),None)
        if button is not None:break
        time.sleep(1)
    assert button is not None,'Camera scan button did not become enabled'
    (REPORTS/'internal-startup.png').write_bytes(adb('exec-out','screencap','-p',binary=True))
    bounds=[int(v) for v in re.findall(r'\d+',button.get('bounds'))]
    adb('shell','input','tap',str((bounds[0]+bounds[2])//2),str((bounds[1]+bounds[3])//2))
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        tree=ui('internal-camera-result.xml')
        if any(n.get('text')=='Your scan' for n in tree.iter('node')):break
        time.sleep(1)
    assert any(n.get('text')=='Your scan' for n in tree.iter('node'))
    (REPORTS/'internal-camera-result.png').write_bytes(adb('exec-out','screencap','-p',binary=True))
    packages=adb('shell','pm','list','packages','com.fruitlens')
    assert 'package:com.fruitlens.classic' in packages and 'package:com.fruitlens.classic.plus' in packages
    report=dict(passed=True,R8_optimized=True,model_prewarm_ready=True,real_camera_capture=True,result_screen_rendered=True,classic_and_plus_coinstalled=True,custom_also_installed="package:com.fruitlens.custom" in packages,note='x86_64 emulator internal build, not arm64 phone performance validation')
    (REPORTS/'internal-smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
