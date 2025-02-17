import requests
response = requests.get("http://localhost:4000/v1/models")
print(response.json())