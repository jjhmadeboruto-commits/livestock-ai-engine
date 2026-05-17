import zipfile
from pathlib import Path
wheel = Path('mediapipe-0.10.35-py3-none-win_amd64.whl')
with zipfile.ZipFile(wheel) as z:
    files = z.namelist()
    for f in sorted(files):
        if 'mediapipe/python' in f or 'mediapipe/__init__.py' in f or 'mediapipe/solutions' in f or 'mediapipe/python/solutions' in f:
            print(f)
