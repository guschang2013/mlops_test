import pandas as pd
import pickle

features_path = "features/iris_features.csv"
target_path = "data/iris.csv"
model_path = "models/iris_model.pkl"

# Load features, target, and model
features = pd.read_csv(features_path)
target = pd.read_csv(target_path, header=None)[4]

# load the model
with open(model_path, "rb") as f:
    model = pickle.load(f)
# model = xgb.Booster(model_file=model_path)

# Evaluate the model
predictions = model.predict(features)
accuracy = (predictions == target).sum() / len(target)
print("Longitud: ", len(target))
print("Longitud: ", len(target))
print("Accuracy:", accuracy)
