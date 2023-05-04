import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)


model_path = "models/iris_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["POST"])
def predict():
    data = request.json["data"]
    features = [[data["sepal_length"], data["sepal_width"], data["petal_length"], data["petal_width"]]]
    prediction = model.predict(features)
   
