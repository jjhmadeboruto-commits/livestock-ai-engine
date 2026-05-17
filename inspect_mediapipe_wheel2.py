import zipfile
from pathlib import Path
wheel = Path('mediapipe-0.10.35-py3-none-win_amd64.whl')
with zipfile.ZipFile(wheel) as z:
    files = sorted(z.namelist())
    print('TOTAL', len(files))
    for f in files:
        if f.startswith('mediapipe/'):
            print(f)
