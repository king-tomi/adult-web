import requests

url = "http://localhost:500/"

r = requests.post(url)

print(r.json())