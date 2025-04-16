<hr></hr>
<h1>Spam Message Detection </h1><br>
This project is a machine learning application that detects whether a given message is spam or not. It uses a trained model and provides a user interface for predictions. <br>
Here is a step-by-step guide for your project:

---

### **1. Create the Model**
Use the following steps to create and save the spam detection model:

1. **Prepare the Dataset**:
   - Load the dataset (`spam.tsv`) and preprocess it.
   - Balance the dataset if necessary.

2. **Train the Model**:
   - Use `TfidfVectorizer` for text vectorization.
   - Train a `RandomForestClassifier` on the data.

3. **Save the Model**:
   - Save the trained model using `pickle`.

Example code to create and save the model:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load dataset
df = pd.read_csv('spam.tsv', sep='\t')

# Balance dataset
ham = df[df['label'] == 'ham']
spam = df[df['label'] == 'spam']
ham = ham.sample(spam.shape[0])
data = pd.concat([ham, spam], axis=0)

# Split data
x_train, x_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=0)

# Train model
pipeline = Pipeline([
    ('TfidfVectorizer', TfidfVectorizer()),
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=100, n_jobs=-1))
])
pipeline.fit(x_train, y_train)

# Save model
pickle.dump(pipeline, open('spam_model.pkl', 'wb'))
```

---

### **2. Run Locally Using Flask API**
Create a Flask API to serve the model.

1. **Flask API Code**:
   - Load the saved model.
   - Create endpoints for prediction.

Example Flask API code:
```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('spam_model.pkl', 'rb'))

@app.route('/')
def home():
    return "Spam Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    if not message:
        return jsonify({"error": "Message field is required"}), 400
    prediction = model.predict([message])[0]
    return jsonify({"message": message, "prediction": "spam" if prediction == "spam" else "not spam"})

if __name__ == '__main__':
    app.run(debug=True)
```

2. **Run the Flask API**:
   ```bash
   python SpamDetectionService.py
   ```

3. **Test Using Postman**:
   - Set the URL to `http://127.0.0.1:5000/predict`.
   - Use the `POST` method.
   - Add a JSON body:
     ```json
     {
       "message": "Congratulations! You have won a lottery."
     }
     ```

---

### **3. Create Docker Image**
1. **Create a `Dockerfile`**:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY . /app
   RUN pip install --no-cache-dir -r requirements.txt
   EXPOSE 8080
   CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "SpamDetectionService:app"]
   ```

2. **Build the Docker Image**:
   ```bash
   docker build -t spam-detector-app .
   ```

3. **Run the Docker Container**:
   ```bash
   docker run -p 8080:8080 spam-detector-app
   ```

---

### **4. Test Using Postman (Docker)**
- Set the URL to `http://127.0.0.1:8080/predict`.
- Use the `POST` method.
- Add a JSON body:
  ```json
  {
    "message": "Your OTP is 1234."
  }
  ```

You will receive a response with the prediction.

Some Screenshots

![WhatsApp Image 2025-04-12 at 23 08 40_8a46268d](https://github.com/user-attachments/assets/0e0c9f5f-9457-4b42-91c1-ae6c5726d5fe)
![WhatsApp Image 2025-04-12 at 23 08 52_02e1312c](https://github.com/user-attachments/assets/0928ccee-b246-4bcf-b4a9-9315dd6c0a30)


![WhatsApp Image 2025-04-14 at 18 17 52_60f407ee](https://github.com/user-attachments/assets/98dcceb8-f1be-4b5c-be28-6904ad0e5758)

![image](https://github.com/user-attachments/assets/80122c04-5bac-412a-b39a-4ce3b9740b76)





Building Docker Image and running the app


![image](https://github.com/user-attachments/assets/9f9b784d-2aa0-4a44-bdf9-5dfb0ff4417e)


Deploying Docker in AWS ECS
![image](https://github.com/user-attachments/assets/b9f47eaa-cbdc-4d70-8e0c-562f279195d6)

