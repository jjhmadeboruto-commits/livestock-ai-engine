import requests

url = 'http://127.0.0.1:5000/api/estimate-weight'
image_path = r'C:\Users\User 2\Downloads\istockphoto-496397741-612x612.jpg'

with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(url, files=files)

print('Status:', response.status_code)
try:
    print('Response:', response.json())
except Exception as e:
    print('Raw response:', response.text)
