from flask import Flask, request, jsonify
import joblib
import numpy as np

# initialize flask app
app = Flask(__name__)

# load trained model
model = joblib.load("model/fraud_model.pkl")

# home route
@app.route("/")
def home():
    return "Credit Card Fraud Detection API Running"


# prediction route
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["features"]

    # convert input to numpy array
    features = np.array(data).reshape(1, -1)

    prediction = model.predict(features)

    result = int(prediction[0])

    if result == 1:
        output = "Fraud Transaction"
    else:
        output = "Normal Transaction"

    return jsonify({"prediction": output})


# run app
if __name__ == "__main__":
    app.run(debug=True)