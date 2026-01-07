import streamlit as st
import pickle
import helper
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))

st.set_page_config(
    page_title="Quora Question Pair Problem",
    page_icon="❓",
    layout="centered"
)


st.title("❓ Quora Question Pair Problem")
st.subheader("Duplicate Question Detection using NLP & Machine Learning")

st.markdown("""
This application predicts whether **two questions have the same meaning**
(i.e. are *duplicates*), inspired by the famous **Quora Question Pairs dataset**.

""")


st.markdown("### ✏️ Enter Questions")

q1 = st.text_area("Question 1", placeholder="e.g. How can I learn machine learning?")
q2 = st.text_area("Question 2", placeholder="e.g. What is the best way to learn ML?")


if st.button("🔎 Check Duplicate"):
    if q1.strip() == "" or q2.strip() == "":
        st.warning("⚠️ Please enter both questions.")
    else:
        with st.spinner("Analyzing questions..."):
            query = helper.query_point_creator(q1, q2)
            prediction = model.predict(query)[0]

        st.markdown("---")

        if prediction == 1:
            st.success("✅ These questions are **Duplicate** (Same Meaning)")
        else:
            st.error("❌ These questions are **Not Duplicate**")


st.markdown("""
---
### 🛠️ Tech Stack Used
- **Python**
- **Natural Language Processing (NLP)**
- **Feature Engineering**
- **Machine Learning**
- **Streamlit for Deployment**

This project follows **industry-level ML workflow** including:
- Data preprocessing
- Feature extraction
- Model training & evaluation
- Model serialization
- Web deployment
""")


st.markdown("""
---
<center>
<b>Made by Vansh</b>  
<br>
<i>Learning ML-NLP by building real-world projects</i>
</center>
""", unsafe_allow_html=True)
