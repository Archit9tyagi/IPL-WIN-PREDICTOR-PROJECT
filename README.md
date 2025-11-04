# IPL-WIN-PREDICTOR-PROJECT
🏏 Machine Learning model to predict IPL match win probabilities using historical data, real-time match inputs, and logistic regression — with an interactive web app built using Streamlit.
# 🏏 IPL Win Predictor (Machine Learning Project)

This project predicts the winning probability of IPL teams based on historical match data and real-time inputs such as score, wickets, and overs.

---

## 🚩 Overview
The **IPL Win Predictor** uses a trained machine learning model to estimate the likelihood of a team winning a match at any given point. It combines data preprocessing, feature engineering, and a logistic regression model for prediction.

---

## ⚙️ Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn, Streamlit
- **Visualization:** Matplotlib, Seaborn
- **Frontend (optional):** Streamlit or Flask

---

## 🧩 Project Structure
- `data/` → Raw and processed datasets  
- `notebooks/` → Jupyter notebooks for EDA and model training  
- `src/` → Python scripts for data cleaning, feature engineering, and model building  
- `web_app/` → Flask or Streamlit app files  
- `requirements.txt` → Python dependencies

---

## 📊 Model
The model is trained on IPL datasets (2008–2023) and predicts win probability using:
- Current score  
- Overs completed  
- Wickets fallen  
- Target score  
- Batting and bowling team  
- Venue  

Algorithm used: **Logistic Regression / Random Forest (tunable)**

---

## 🖥️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/IPL-Win-Predictor.git
   cd IPL-Win-Predictor
