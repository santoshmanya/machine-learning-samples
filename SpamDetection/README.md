<hr></hr>
Spam Message Detection <br>
This project is a machine learning application that detects whether a given message is spam or not. It uses a trained model and provides a user interface for predictions. <br>


Features <br>
Spam Detection: Classifies messages as spam or not spam. <br>
Streamlit Web App: A user-friendly interface for inputting messages and viewing predictions. <br>
Jupyter Notebook: Includes data preprocessing, feature engineering, and model training steps. <br>
TensorFlow Integration: Utilized for testing TensorFlow setup and compatibility. <br>
Installation <br>
Prerequisites <br>
Python 3.8, 3.9, or 3.10 (TensorFlow is not compatible with Python 3.13) <br>
pip (latest version recommended) <br>
Steps <br>
Clone the repository: <br>


git clone https://github.com/santoshmanya/machine-learning-samples.git <br>
Create a virtual environment: <br>


python -m venv venv <br>
.\venv\Scripts\activate  # On Windows <br>
Install dependencies: <br>


pip install -r requirements.txt <br>
Set the environment variable (optional): <br>


set TF_ENABLE_ONEDNN_OPTS=0  # On Windows <br>
Usage <br>
Running the Streamlit App <br>
Start the app: <br>
streamlit run Web/SpamMessageDetection.py <br>
Open the provided URL in your browser. <br>
Running TensorFlow Test <br>
Execute the script to verify TensorFlow installation: <br>
python Util/TensorflowTest.py <br>
Jupyter Notebook <br>
Open the notebook in PyCharm or Jupyter: <br>
jupyter notebook SpamMessageDetection.ipynb <br>
Project Structure <br>
Web/SpamMessageDetection.py: Streamlit app for spam detection. <br>
SpamMessageDetection.ipynb: Jupyter Notebook for data preprocessing and model training. <br>
Util/TensorflowTest.py: Script to test TensorFlow installation. <br>
spam_model.pkl: Pre-trained model for predictions. <br>
Dependencies <br>
TensorFlow (optional)<br>
Streamlit <br>
Scikit-learn <br>
Pandas <br>
Matplotlib <br>
Numpy <br>
License <br>
This project is licensed under the MIT License. <br>
![WhatsApp Image 2025-04-12 at 23 08 40_8a46268d](https://github.com/user-attachments/assets/0e0c9f5f-9457-4b42-91c1-ae6c5726d5fe)
![WhatsApp Image 2025-04-12 at 23 08 52_02e1312c](https://github.com/user-attachments/assets/0928ccee-b246-4bcf-b4a9-9315dd6c0a30)


![WhatsApp Image 2025-04-14 at 18 17 52_60f407ee](https://github.com/user-attachments/assets/98dcceb8-f1be-4b5c-be28-6904ad0e5758)

Developed API in flask and called this API using react client.
![image](https://github.com/user-attachments/assets/80122c04-5bac-412a-b39a-4ce3b9740b76)


Running Tensorflow on Windows Native without GPU. Currently tensor flow does not support python 3.13

1. Install python 3.11
2. Install Tensorflow 2.19.0
3. Test Tensorflow Installaiton and if it gives below error then set the environment variable TF_ENABLE_ONEDNN_OPTS to 0
4. ![image](https://github.com/user-attachments/assets/ca1b2f4e-8458-4a3d-8608-e86218f87393)



Building Docker Image and running the app


![image](https://github.com/user-attachments/assets/c09c843e-4337-49a9-98be-5a031b807138)


6. ![image](https://github.com/user-attachments/assets/50cd8438-bc52-43d8-a7be-6fee8e9467bc)
7. After updating environment viarables, you will see below results when you test tensorflow
8. ![image](https://github.com/user-attachments/assets/81b02a56-89f2-4147-8fec-128bb815d70a)
