Sample Projects on ML

Projects 1 : Spam Detection - RandomForestClassifier https://www.ibm.com/think/topics/random-forest

<hr></hr>
Spam Message Detection
This project is a machine learning application that detects whether a given message is spam or not. It uses a trained model and provides a user interface for predictions.


Features
Spam Detection: Classifies messages as spam or not spam.
Streamlit Web App: A user-friendly interface for inputting messages and viewing predictions.
Jupyter Notebook: Includes data preprocessing, feature engineering, and model training steps.
TensorFlow Integration: Utilized for testing TensorFlow setup and compatibility.
Installation
Prerequisites
Python 3.8, 3.9, or 3.10 (TensorFlow is not compatible with Python 3.13)
pip (latest version recommended)
Steps
Clone the repository:


git clone <repository-url>
cd <repository-folder>
Create a virtual environment:


python -m venv venv
.\venv\Scripts\activate  # On Windows
Install dependencies:


pip install -r requirements.txt
Set the environment variable (optional):


set TF_ENABLE_ONEDNN_OPTS=0  # On Windows
Usage
Running the Streamlit App
Start the app:
streamlit run Web/SpamMessageDetection.py
Open the provided URL in your browser.
Running TensorFlow Test
Execute the script to verify TensorFlow installation:
python Util/TensorflowTest.py
Jupyter Notebook
Open the notebook in PyCharm or Jupyter:
jupyter notebook SpamMessageDetection.ipynb
Project Structure
Web/SpamMessageDetection.py: Streamlit app for spam detection.
SpamMessageDetection.ipynb: Jupyter Notebook for data preprocessing and model training.
Util/TensorflowTest.py: Script to test TensorFlow installation.
spam_model.pkl: Pre-trained model for predictions.
Dependencies
TensorFlow
Streamlit
Scikit-learn
Pandas
Matplotlib
Numpy
License
This project is licensed under the MIT License.



Running Tensorflow on Windows Native without GPU. Currently tensor flow does not support python 3.13

1. Install python 3.11
2. Install Tensorflow 2.19.0
3. Test Tensorflow Installaiton and if it gives below error then set the environment variable TF_ENABLE_ONEDNN_OPTS to 0
4. ![image](https://github.com/user-attachments/assets/ca1b2f4e-8458-4a3d-8608-e86218f87393)

5. ![image](https://github.com/user-attachments/assets/50cd8438-bc52-43d8-a7be-6fee8e9467bc)
6. After updating environment viarables, you will see below results when you test tensorflow
7. ![image](https://github.com/user-attachments/assets/81b02a56-89f2-4147-8fec-128bb815d70a)

