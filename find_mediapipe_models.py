import pkgutil, importlib, os
import mediapipe
root = mediapipe.__path__[0]
for base, dirs, files in os.walk(root):
    for f in files:
        if f.endswith(('.tflite', '.pb', '.task', '.lite', '.model', '.txt')):
            if 'pose' in f.lower() or 'landmarker' in f.lower() or 'holistic' in f.lower():
                print(os.path.join(base, f))
