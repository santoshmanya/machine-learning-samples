# pip install -U streamlit
# pip install -u plotly
# pip install -u flask


import streamlit as st
import pickle as pkl
import tensorflow as tf

# Load the model
model = pkl.load(open("spam_model.pkl", "rb"))

st.title("Spam Message Detection")
st.write("This app detects if the message is spam or not.")
message = st.text_input("Enter a message: ")
if st.button("Predict"):
    prediction = model.predict([message])
    print(prediction)
    if prediction[0] == "spam":
        st.error("This message is spam.")
    else:
        st.success("This message is not spam.")
