"""Verify packaged models, signatures, alignment, identity and actual test evidence."""
from pathlib import Path
import json,hashlib,subprocess,zipfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/02-fruit-lens-custom'
REPORTS=APP/'verification'
SDK=ROOT/'android/.tools/android-sdk/build-tools/36.0.0'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    apk=APP/'apk/Fruit-Lens-Custom-v1.3-arm64.apk'
    bundle=json.loads((APP/'MODEL_BUNDLE.json').read_text())
    with zipfile.ZipFile(apk) as z:
        for name,record in bundle['files'].items():
            assert hashlib.sha256(z.read('assets/'+name)).hexdigest()==record['sha256'],name
            assert sha(APP/'android/app/src/main/assets'/name)==record['sha256']
    for tool,args,filename in [
        ('aapt2.exe',['dump','badging',str(apk)],'apk-badging.txt'),
        ('apksigner.bat',['verify','--verbose','--print-certs',str(apk)],'apk-signature.txt'),
        ('zipalign.exe',['-c','-P','16','-v','4',str(apk)],'apk-alignment.txt')]:
        p=subprocess.run([str(SDK/tool),*args],capture_output=True,text=True,check=True)
        (REPORTS/filename).write_text(p.stdout+p.stderr,encoding='utf-8')
    badging=(REPORTS/'apk-badging.txt').read_text()
    assert "name='com.fruitlens.custom'" in badging
    assert "versionName='1.3.0-custom'" in badging and "versionCode='4'" in badging
    assert "application-label:'Fruit Lens Custom'" in badging
    assert "minSdkVersion:'26'" in badging
    assert 'android.permission.INTERNET' not in badging
    assert "native-code: 'arm64-v8a'" in badging
    assert 'OK (6 tests)' in (REPORTS/'instrumentation.txt').read_text()
    assert json.loads((REPORTS/'internal-smoke.json').read_text())['passed']
    tree=ET.parse(APP/'android/app/build/test-results/testDebugUnitTest/TEST-com.fruitlens.app.ScoreMathTest.xml').getroot()
    assert tree.get('failures')=='0' and tree.get('errors')=='0'
    display=ET.parse(APP/'android/app/build/test-results/testDebugUnitTest/TEST-com.fruitlens.app.FreshnessDisplayTest.xml').getroot()
    assert display.get('failures')=='0' and display.get('errors')=='0'
    import sys
    p=subprocess.run([sys.executable,str(ROOT/'apps/01-fruit-lens-classic/verify_snapshot.py')],capture_output=True,text=True,check=True)
    (REPORTS/'classic-preservation.txt').write_text(p.stdout)
    for entry in json.loads((ROOT/'models/catalog.json').read_text()):
        for name,record in entry['files'].items():assert sha(ROOT/'models'/entry['version']/name)==record['sha256']
    metadata=dict(edition='Fruit Lens Custom',applicationId='com.fruitlens.custom',sourceNamespace='com.fruitlens.app',versionName='1.3.0-custom',versionCode=4,detector='yolov8n-fruit9-v2',classifier='mobilenetv3small-fruit9-v1',inference_backend='CPU',threads_per_model=2,GPU=False,NPU=False,apk=dict(path=apk.relative_to(APP).as_posix(),bytes=apk.stat().st_size,sha256=sha(apk),signing='local development key; verified, not Play Store signing'),models=bundle['files'],verification=dict(JVM_tests=int(tree.get('tests'))+int(display.get('tests')),Android_tests=6,Android_classifier_parity_cases=18,classic_snapshot_files=71,APK_identity_and_signature=True,APK_16KiB_alignment=True,packaged_model_hashes=True,all_catalog_hashes=True),phone_latency_measured=False,independent_camera_accuracy_validated=False)
    metadata['verification']['R8_internal_startup_and_capture'] = True
    (APP/'VERSION.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (REPORTS/'summary.json').write_text(json.dumps(dict(passed=True,**metadata['verification'],note='Emulator execution and packaging evidence; not production accuracy or real-phone latency validation.'),indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
