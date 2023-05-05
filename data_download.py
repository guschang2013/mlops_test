import os
import requests

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
output_path = "./data/iris.csv"

if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))

response = requests.get(url)
response.raise_for_status()

with open(output_path, 'wb') as f:
    f.write(response.content)
