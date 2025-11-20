import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- CUSTOM CSS ----------------------
page_bg = """
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
}

h1, h2, h3 {
    font-weight: 800 !important;
}

.css-1d391kg {
    color: white !important;
}

/* Cards */
.card {
    padding: 25px;
    background: rgba(255, 255, 255, 0.10);
    border-radius: 15px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.2);
}

input, select {
    color: black !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("<h1 style='text-align:center;'>🏏 IPL MATCH WIN PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; opacity:0.8;'>Powered by Machine Learning</h3><br>", unsafe_allow_html=True)

# ---------------------- DATA ----------------------
teams = sorted([
    'Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
    'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
    'Rajasthan Royals','Delhi Capitals'
])

cities = sorted([
    'Hyderabad','Bangalore','Mumbai','Indore','Kolkata','Delhi','Chandigarh','Jaipur',
    'Chennai','Cape Town','Port Elizfabeth','Durban','Centurion','East London',
    'Johannesburg','Kimberley','Bloemfontein','Ahmedabad','Cuttack','Nagpur','Dharamsala',
    'Visakhapatnam','Pune','Raipur','Ranchi','Abu Dhabi','Sharjah','Mohali','Bengaluru'
])

# ---------------------- MODEL LOAD ----------------------
pipe_path = "pipe.pkl"

if os.path.exists(pipe_path):
    pipe = pickle.load(open(pipe_path, "rb"))
else:
    pipe = None
    st.warning("⚠ Model file 'pipe.pkl' not found. Place it in the same folder.", icon="⚠")

# ---------------------- FUNCTION ----------------------
def overs_to_balls(overs_float):
    overs_int = int(overs_float)
    frac = round((overs_float - overs_int) * 10)
    if frac > 5:
        frac = 5
    return overs_int * 6 + frac

# ---------------------- INPUT UI ----------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Match Inputs")

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox("🏏 Batting Team", teams)
    with col2:
        bowling_team = st.selectbox("🎯 Bowling Team", teams)

    selected_city = st.selectbox("📍 Host City", cities)

    col3, col4 = st.columns(2)
    with col3:
        target = st.number_input("🎯 Target", min_value=0, step=1)
    with col4:
        score = st.number_input("🏃‍♂️ Current Score", min_value=0, step=1)

    col5, col6 = st.columns(2)
    with col5:
        overs = st.number_input("⏳ Overs Completed (e.g. 10.3)", min_value=0.0, step=0.1, format="%.1f")
    with col6:
        wickets_out = st.number_input("💔 Wickets Fallen", min_value=0, max_value=10, step=1)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- PREDICTION BUTTON ----------------------
predict_button = st.button("🔮 Predict Win Probability", use_container_width=True)

# ---------------------- PREDICTION LOGIC ----------------------
if predict_button:

    if pipe is None:
        st.error("❌ Model not loaded. Add 'pipe.pkl' next to app.py")
    else:
        if target <= 0:
            st.error("Please enter a valid target (>0)")
        else:
            balls_done = overs_to_balls(overs)
            balls_left = 120 - balls_done
            
            if balls_left <= 0:
                st.error("❌ Overs exceed 20. Check your input.")
            else:
                runs_left = target - score
                wickets_left = 10 - wickets_out

                # CRR & RRR
                overs_int = int(overs)
                frac = round((overs - overs_int) * 10)
                overs_fraction = overs_int + frac/6

                crr = score / overs_fraction if overs_fraction > 0 else 0
                rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

                # DataFrame
                input_df = pd.DataFrame({
                    'batting_team':[batting_team],
                    'bowling_team':[bowling_team],
                    'city':[selected_city],
                    'runs_left':[runs_left],
                    'balls_left':[balls_left],
                    'wickets':[wickets_left],
                    'total_runs_x':[target],
                    'crr':[crr],
                    'rrr':[rrr]
                })

                # Predict
                result = pipe.predict_proba(input_df)[0]
                loss, win = result[0], result[1]

                # ---------------------- RESULT CARD ----------------------
                st.markdown("<br><div class='card'>", unsafe_allow_html=True)
                st.subheader("📊 Win Probability")

                colA, colB = st.columns(2)
                with colA:
                    st.metric(
                        label=f"{batting_team} Win Chance",
                        value=f"{round(win*100, 2)} %"
                    )
                with colB:
                    st.metric(
                        label=f"{bowling_team} Win Chance",
                        value=f"{round(loss*100, 2)} %"
                    )

                st.markdown("</div>", unsafe_allow_html=True)