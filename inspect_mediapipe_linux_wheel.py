import zipfile
from pathlib import Path
wheel = Path('mediapipe-0.10.35-py3-none-manylinux_2_28_x86_64.whl')
with zipfile.ZipFile(wheel) as z:
    files = sorted(z.namelist())
    print('TOTAL', len(files))
    for f in files:
        if f.startswith('mediapipe/'):
            if 'python' in f or 'solutions' in f or 'tasks' in f or 'pose' in f:
                print(f)
