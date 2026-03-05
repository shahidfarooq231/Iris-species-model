import streamlit as st
import numpy as np
import pickle

with open("iris_dataset.pkl",'rb') as f:
    model = pickle.load(f)

st.title("Iris FLower Prediction")


Sepal_Length=st.slider("sepal length(cm)", 4.0, 8.0)
Sepal_Width=st.slider("sepal width(cm)", 2.0, 4.5)
Petal_Length=st.slider("petal length(cm)", 1.0, 7.0)
Petal_Width=st.slider("petal width(cm)", 0.1, 2.5) 


if st.button("prediction"):
    input_data=np.array([[Sepal_Length,Sepal_Width,Petal_Length,Petal_Width]])
    prediction = model.predict(input_data)
    Species = ['Setosa', 'versicolor', 'Virginica']
    st.success(f"Predicted Iris Species: {Species[prediction[0]]}")