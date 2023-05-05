import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

data_path = "data/iris.csv"
output_path = "features/iris_features.csv"

# Load data
data = pd.read_csv(data_path, header=None)
data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# Define a dictionary to map class names to integers
class_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

# Replace class names with integer values
data["class"] = data["class"].map(class_map)

# Get the features
features = data.drop(columns=["class"])

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))
    
# Save the features
pd.DataFrame(scaled_features).to_csv(output_path, index=False)
