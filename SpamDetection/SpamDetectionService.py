from flask import Flask, request, jsonify
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open("spam_model.pkl", "rb"))

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