import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os
import pickle

features_path = "features/iris_features.csv"
target_path = "data/iris.csv"
output_path = "models/iris_model.pkl"

# Load features and target
features = pd.read_csv(features_path)
target = pd.read_csv(target_path, header=None)[4]

le = LabelEncoder()
target = le.fit_transform(target)

# Train the model
model = xgb.XGBClassifier()
model.fit(features, target)

if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))

# Save the model
# xgb.save_model(output_path, model)
with open(output_path, "wb") as f:
    pickle.dump(model, f)
