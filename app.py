import streamlit as st
import joblib
import pandas as pd

st.title("My ML App")

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

f1 = st.slider("Feature 1 (Age)", 18, 60, 25)
f2 = st.slider("Feature 2 (Income)", 10000, 100000, 40000)
f3 = st.selectbox("Feature 3", [0, 1, 2])

if st.button("Predict"):
    res = model.predict(pd.DataFrame([[f1, f2, f3]]))
    st.success(f"Result: {res[0]}")
