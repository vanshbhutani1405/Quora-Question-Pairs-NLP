# 🔁 Quora Question Pairs – Duplicate Detection using NLP & ML

This project solves a real-world NLP problem:  
**Given two questions from Quora, can we predict whether they mean the same thing?**

Quora faces this issue daily — users often ask the same question in different ways.  
Detecting duplicate questions helps:
- Reduce clutter  
- Improve search quality  
- Save time for users  

This project builds an end-to-end Machine Learning + NLP system to solve exactly that.

---

## 🚀 Project Overview

The goal is to classify whether two questions are **Duplicate** or **Not Duplicate**.

### What this project includes:
- Text cleaning & preprocessing  
- Feature engineering (15+ similarity features)  
- Text vectorization (TF-IDF & Count Vectorizer)  
- Multiple ML models  
- Model evaluation using Accuracy & F1-score  
- Hyperparameter tuning  
- Final deployment using Streamlit  

---

## 🧠 Techniques Used

### 🔹 Text Preprocessing
- Lowercasing  
- Removing HTML tags  
- Removing special characters  
- Expanding contractions  
- Tokenization  

### 🔹 Feature Engineering (15+ Features)
Some of the engineered features:
- Length difference  
- Word overlap ratio  
- Common words count  
- First & last word match  
- Longest common substring ratio  
- Fuzzy matching scores  
  - Fuzz ratio  
  - Partial ratio  
  - Token sort ratio  
  - Token set ratio  

### 🔹 Text Vectorization
- TF-IDF Vectorizer  
- Count Vectorizer (Bag of Words)  

### 🔹 Models Trained
- Naive Bayes  
- Logistic Regression  
- Linear SVM  
- Random Forest  
- XGBoost (Best performing model)  

---

## 📊 Results

| Vectorizer        | Accuracy | F1 Score |
|-------------------|----------|----------|
| TF-IDF            | ~79%     | ~0.703   |
| Count Vectorizer  | ~78.5%   | ~0.72    |

- Best model: **XGBoost**
- Hyperparameter tuning tried, but no major improvement over baseline.

---

## 🌐 Deployment

The final model is deployed using **Streamlit** for real-time predictions.

### 🔗 Live App  
👉 https://quora-question-pairs-nlp.streamlit.app/

### 🔗 GitHub Repository  
👉 https://github.com/vanshbhutani1405/Quora-Question-Pairs-NLP

---

## 🗂️ Project Structure

