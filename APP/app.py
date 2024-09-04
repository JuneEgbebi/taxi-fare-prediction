from flask import Flask, request, jsonify
from joblib import load
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

linreg_model = load('gradient_boosting_model.joblib')  # Load your Linear Regression model

@app.route('/predict', methods=['POST'])
def prediction():
    data = request.get_json()  # Get JSON data from the request
    data_array = np.array(data).reshape(1, -1)  # Ensure the input is a NumPy array and reshape if necessary
    predictions = linreg_model.predict(data_array)  # Make predictions
    return jsonify({
        "prediction": predictions.tolist()  # Return predictions as a JSON list
    })

if __name__ == '__main__':
    app.run(debug=True)

