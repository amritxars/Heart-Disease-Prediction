import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("❤️ Heart Disease Prediction")

# Input fields
age = st.number_input("Age")
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["typical angina","atypical angina","non-anginal","asymptomatic"])
trestbps = st.number_input("Resting BP")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar >120", ["TRUE","FALSE"])
restecg = st.selectbox("Rest ECG", ["normal","lv hypertrophy"])
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Angina", ["TRUE","FALSE"])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope", ["upsloping","flat","downsloping"])
ca = st.number_input("CA")
thal = st.selectbox("Thal", ["normal","fixed defect","reversable defect"])

# Convert input
sex = 1 if sex=="Male" else 0
fbs = 1 if fbs=="TRUE" else 0
exang = 1 if exang=="TRUE" else 0

cp_map = {"typical angina":0,"atypical angina":1,"non-anginal":2,"asymptomatic":3}
restecg_map = {"normal":0,"lv hypertrophy":1}
slope_map = {"upsloping":0,"flat":1,"downsloping":2}
thal_map = {"normal":1,"fixed defect":2,"reversable defect":3}

cp = cp_map[cp]
restecg = restecg_map[restecg]
slope = slope_map[slope]
thal = thal_map[thal]

# Dummy training (replace later with saved model)
df = pd.read_csv("heart.csv")
df['sex'] = df['sex'].map({'Male':1,'Female':0})
df['fbs'] = df['fbs'].map({'TRUE':1,'FALSE':0})
df['exang'] = df['exang'].map({'TRUE':1,'FALSE':0})
df['cp'] = df['cp'].map(cp_map)
df['restecg'] = df['restecg'].map(restecg_map)
df['slope'] = df['slope'].map(slope_map)
df['thal'] = df['thal'].map(thal_map)
df = df.rename(columns={'num':'target'})
df = df.drop(['id','dataset'], axis=1)

X = df.drop('target', axis=1)
y = df['target']

model = RandomForestClassifier()
model.fit(X,y)

# Prediction
if st.button("Predict"):
    input_data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    pred = model.predict(input_data)

    if pred[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")