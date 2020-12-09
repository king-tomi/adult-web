import requests

url = "http://localhost:500/results"

r = requests.post(url)

print(r.json())