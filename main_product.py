import streamlit as st
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the model
lo_R = LogisticRegression(solver='lbfgs',max_iter=120)
df = pd.read_csv('heart.csv')
X = df.iloc[:, :-1]
Y = df.iloc[:, -1:]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
lo_R.fit(x_train, y_train)

# Define the form
st.header('Heart Disease Prediction')


# Define the form inputs
col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Enter your Age', step=1)
    sex = st.number_input('Enter your Gender', step=1)
    cp = st.number_input('Enter your Chest Pain (0 & 1)', step=1)
    trestbps = st.number_input('Enter your Fasting Blood Sugar', step=1)
    chol = st.number_input('Enter your Serum Cholesterol', step=1)
    fbs = st.number_input('Enter your FBS', step=1)
    restecg = st.number_input('Enter your Resting Electrocardiographic Results', step=1)

with col2:
    thalach = st.number_input('Enter your thalach', step=1)
    exang = st.number_input('Enter your Exercise Induced Angina', step=1)
    oldpeak = st.number_input('Enter your oldpeak', step=0.1)
    slope = st.number_input('Enter your slope', step=1)
    ca = st.number_input('Enter your Number of Major Vessels', step=1)
    thal = st.number_input('Enter your Thalassemia', step=1)

# Gather the data
data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Make prediction on button click
if st.button('Predict'):
    prediction = lo_R.predict([data]) 
    if prediction == 1:
        st.write("Yes, you have a more chances of heart atteck.")
    else:
        st.write("No, you don't have heart disease.")
