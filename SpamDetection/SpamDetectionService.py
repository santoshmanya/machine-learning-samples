from flask import Flask, request, jsonify
from flask_cors import CORS # 1. Import CORS
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app) # 2. Initialize CORS with your app. This allows all origins by default.



# Load the pre-trained model
model = pickle.load(open("spam_model.pkl", "rb"))

from flask import Flask, request, jsonify
from flask_cors import CORS # 1. Import CORS
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app) # 2. Initialize CORS with your app. This allows all origins by default.



# Load the pre-trained model
model = pickle.load(open("spam_model.pkl", "rb"))

@app.route('/info')
def info():
    """
    Endpoint to check if the service is running.
    """
    return "{'name' :  'Spam Detection Service', 'version' : '1.0', 'status' : 'running'}"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict if a message is spam or not.
    """
    try:
        # Get the JSON data from the request
        data = request.get_json()
        message = data.get('message', '')

        # Validate input
        if not message:
            return jsonify({"error": "Message field is required"}), 400

        # Make prediction
        prediction = model.predict([message])[0]

        # Return the result
        result = {
            "message": message,
            "prediction": "spam" if prediction == "spam" else "not spam"
        }
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
    
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict if a message is spam or not.
    """
    try:
        # Get the JSON data from the request
        data = request.get_json()
        message = data.get('message', '')

        # Validate input
        if not message:
            return jsonify({"error": "Message field is required"}), 400

        # Make prediction
        prediction = model.predict([message])[0]

        # Return the result
        result = {
            "message": message,
            "prediction": "spam" if prediction == "spam" else "not spam"
        }
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
