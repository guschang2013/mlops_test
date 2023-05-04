import requests

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
output_path = "data/iris.csv"

response = requests.get(url)
response.raise_for_status()

with open(output_path, 'wb') as f:
    f.write(response.content)
