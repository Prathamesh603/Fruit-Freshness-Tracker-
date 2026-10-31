"""Download public sources without touching the existing app/training folders."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    partial = path.with_suffix(path.suffix + '.partial')
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'FruitLens-dataset-preparation/1.0'})
            with urllib.request.urlopen(req, timeout=60) as response, partial.open('wb') as out:
                expected = response.headers.get('Content-Length')
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if expected and partial.stat().st_size != int(expected):
                raise ValueError('Incomplete download')
            partial.replace(path)
            return path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    jobs = {
        'fruit16k': ('https://data.mendeley.com/public-api/zip/6ps7gtp2wg/download/1',
                     ROOT / 'raw/fruit16k/publisher.zip'),
        'detection_boxes': ('https://codeload.github.com/zijianchen98/Fruit-freshness-detection-dataset/zip/0cdcf30cb86a6a051b666d9138b5436264972040',
                            ROOT / 'raw/fruit_freshness_boxes/publisher.zip'),
        'openimages_classes': ('https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv',
                               ROOT / 'sources/openimages/class-descriptions-boxable.csv'),
        'openimages_validation_boxes': ('https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',
                                        ROOT / 'sources/openimages/validation-annotations-bbox.csv'),
        'openimages_validation_info': ('https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv',
                                       ROOT / 'sources/openimages/validation-images-with-rotation.csv'),
    }
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download, url, path): (name, url, path)
                   for name, (url, path) in jobs.items()}
        for future in concurrent.futures.as_completed(futures):
            name, url, path = futures[future]
            try:
                future.result()
                digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(path.read_bytes()).hexdigest()
                results[name] = {'url': url, 'path': path.relative_to(ROOT).as_posix(),
                                 'bytes': path.stat().st_size, 'sha256': digest, 'status': 'downloaded'}
                print(name, 'downloaded', path.stat().st_size, flush=True)
            except Exception as error:
                results[name] = {'url': url, 'status': 'failed', 'error': str(error)}
                print(name, 'FAILED', str(error), flush=True)
            dest = ROOT / 'reports/downloads.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
