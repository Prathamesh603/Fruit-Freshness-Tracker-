"""Finish evaluation/reporting after this run completes, without launching new training."""
from pathlib import Path
import subprocess,sys,time,json

RUN=Path(__file__).resolve().parent


def main():
    deadline=time.monotonic()+8*3600
    summary=RUN/'reports/training_summary.json'
    while True:
        if summary.exists():
            try:
                if json.loads(summary.read_text())['training_completed']:break
            except (ValueError,KeyError):pass
        if time.monotonic()>deadline:raise TimeoutError('Training completion was not recorded')
        log=RUN/'reports/train_console.log'
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-4000));tail=f.read().decode(errors='replace')
            if 'Traceback (most recent call last):' in tail:
                raise RuntimeError('Training log contains a terminal error; inspect it before evaluating')
        time.sleep(5)
    for script in ['evaluate_detector.py','finalize_report.py','verify_experiment.py']:
        print('RUNNING',script,flush=True)
        (RUN/'reports/pipeline_progress.json').write_text(json.dumps({'stage':script,'completed':False},indent=2))
        with (RUN/'reports'/f'{Path(script).stem}_console.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u',str(RUN/script)],stdout=log,stderr=subprocess.STDOUT,cwd=RUN.parents[1])
        if result.returncode:raise RuntimeError(script+' failed; inspect its console log')
    (RUN/'reports/pipeline_progress.json').write_text(json.dumps({'stage':'complete','completed':True},indent=2))
    print('TRAINING_EVALUATION_EXPORT_AND_VERIFICATION_COMPLETE',flush=True)


if __name__=='__main__':main()
