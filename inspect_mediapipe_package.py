import importlib, pkgutil, os
import mediapipe as mp
print('mp file:', mp.__file__)
print('has solutions:', hasattr(mp, 'solutions'))
print('dir solutions candidates:', [n for n in dir(mp) if 'sol' in n.lower()])
print('submodules root:', [m.name for m in pkgutil.iter_modules(mp.__path__)])
for candidate in ['mediapipe.python', 'mediapipe.python.solutions', 'mediapipe.tasks.python.vision.pose_landmarker']:
    try:
        mod = importlib.import_module(candidate)
        print('IMPORT OK:', candidate, '->', getattr(mod, '__file__', 'builtin'))
        print('  attrs:', [n for n in dir(mod) if 'Pose' in n or 'solution' in n.lower() or 'Landmarker' in n])
    except Exception as e:
        print('IMPORT FAIL:', candidate, e)
try:
    import inspect
    import mediapipe.tasks.python.vision.pose_landmarker as pl
    print('pose_landmarker file:', pl.__file__)
    source = inspect.getsource(pl)
    print(source[:8000])
except Exception as e:
    print('SOURCE FAIL', e)
