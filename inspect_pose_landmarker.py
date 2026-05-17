import importlib, inspect
mod = importlib.import_module('mediapipe.tasks.python.vision.pose_landmarker')
print('class file:', mod.__file__)
print(inspect.getsource(mod.PoseLandmarker)[:12000])
