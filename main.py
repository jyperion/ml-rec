from flask import Flask, jsonify, request
from typing import Literal
import joblib


app = Flask(__name__)


LABELS = Literal[
    "Dementia",
    "ALS",
    "Obsessive Compulsive Disorder",
    "Scoliosis",
    "Parkinson’s Disease",
]

# Load the trained model and vectorizer
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def predict(description: str) -> dict:
    """
    Updated predict function to return both label and confidence.
    If the highest probability is below 0.6, returns 'unsure'.
    """
    description_tfidf = vectorizer.transform([description])
    pred_label = model.predict(description_tfidf)[0]
    probabilities = model.predict_proba(description_tfidf)[0]
    confidence = float(max(probabilities))
    if confidence < 0.60:
        return {"prediction": "unsure", "confidence": confidence}
    return {"prediction": pred_label, "confidence": confidence}


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/predict", methods=["POST"])
def identify_condition():
    data = request.get_json(force=True)
    result = predict(data["description"])
    return jsonify(result)


if __name__ == "__main__":
    app.run()