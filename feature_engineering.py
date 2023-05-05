import pandas as pd
from sklearn.preprocessing import StandardScaler

data_path = "data/iris.csv"
output_path = "features/iris_features.csv"

# Load data
data = pd.read_csv(data_path, header=None)
data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# Perform feature engineering
features = data.drop(columns=["class"])

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Save the features
pd.DataFrame(scaled_features).to_csv(output_path, index=False)
