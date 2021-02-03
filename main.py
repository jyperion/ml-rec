from flask import Flask, jsonify, request
from typing import Literal


app = Flask(__name__)


LABELS = Literal[
    "Dementia",
    "ALS",
    "Obsessive Compulsive Disorder",
    "Scoliosis",
    "Parkinson’s Disease",
]


def predict(description: str) -> LABELS:
    """
    Function that should take in the description text and return the prediction
    for the class that we identify it to.
    The possible classes are: ['Dementia', 'ALS',
                                'Obsessive Compulsive Disorder',
                                'Scoliosis', 'Parkinson’s Disease']
    """
    raise NotImplementedError()


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/predict", methods=["POST"])
def identify_condition():
    data = request.get_json(force=True)

    prediction = predict(data["description"])

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run()