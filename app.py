from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
with open("model/house_price_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["overallqual"]),
            float(request.form["grlivarea"]),
            float(request.form["totalbsmtsf"]),
            float(request.form["garagecars"]),
            float(request.form["fullbath"]),
            float(request.form["yearbuilt"])
        ]

        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
