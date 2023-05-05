import requests

url = "http://localhost:5000/predict"
data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

response = requests.post(url, json=data)

if response.ok:
    print(response.json())
else:
    print("Request failed with status code", response.status_code)
