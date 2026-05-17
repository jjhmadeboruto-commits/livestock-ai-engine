import importlib, inspect
pl = importlib.import_module('mediapipe.tasks.python.vision.pose_landmarker')
print('PoseLandmarkerOptions:', pl.PoseLandmarkerOptions)
print(inspect.getsource(pl.PoseLandmarkerOptions)[:12000])
