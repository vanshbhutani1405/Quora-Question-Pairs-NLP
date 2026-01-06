import os
import pickle
import streamlit as st
import helper

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open(MODEL_PATH, "rb"))

# ---------------- UI ----------------
st.set_page_config(page_title="Duplicate Question Detector", layout="centered")

st.title("🔁 Duplicate Question Detector")
st.write("Check whether two questions are semantically duplicate using ML + NLP")

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

THRESHOLD = 0.75  # decision threshold

if st.button("Check"):
    if q1.strip() == "" or q2.strip() == "":
        st.warning("Please enter both questions.")
    else:
        query = helper.query_point_creator(q1, q2)

        # Probability of being duplicate
        proba = model.predict_proba(query)[0][1]

        st.write(f"Duplicate probability: **{proba:.2f}**")

        if proba >= THRESHOLD:
            st.success(f"✅ Duplicate (confidence: {proba:.2f})")
        else:
            st.warning(f"❌ Not Duplicate (confidence: {proba:.2f})")
