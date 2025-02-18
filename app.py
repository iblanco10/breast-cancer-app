import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load trained Logistic Regression model and Scaler
model_path = "logistic_regression.pkl"
scaler_path = "scaler.pkl"

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Home route - renders the input form
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        input_features = [float(request.form[key]) for key in request.form.keys()]
        
        # Convert to NumPy array and reshape for model input
        input_data = np.array(input_features).reshape(1, -1)
        
        # Scale the input data using the saved scaler
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]

        # Convert prediction output to meaningful label
        result = "Benign (2)" if prediction == 2 else "Malignant (4)"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
