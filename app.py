import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

# Load model pipe safely
pipe_path = 'pipe.pkl'
if os.path.exists(pipe_path):
    pipe = pickle.load(open(pipe_path, 'rb'))
else:
    pipe = None
    st.warning(f"Model file not found at '{pipe_path}'. Prediction will not work until you place the file there.")

st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=0, step=1, format="%d")

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, step=1, format="%d")
with col4:
    # Allow float like 10.3 (10 overs and 3 balls)
    overs = st.number_input('Overs completed (e.g. 10 or 10.3 where .3 = 3 balls)', min_value=0.0, step=0.1, format="%.1f")
with col5:
    wickets_out = st.number_input('Wickets out', min_value=0, max_value=10, step=1, format="%d")

def overs_to_balls(overs_float):
    """
    Convert overs (possibly decimal-like where fractional digit = balls) to total balls completed.
    Examples:
      10.0 -> 60
      10.3 -> 63 (10 overs and 3 balls)
    If fractional part looks >5 (invalid), it will be capped to 5.
    """
    overs_int = int(overs_float)
    frac = round((overs_float - overs_int) * 10)  # expects decimal like .3 means 3 balls
    if frac < 0:
        frac = 0
    if frac > 5:
        # If user passed something like 10.7, cap to 5 and add warning later
        frac = 5
    return overs_int * 6 + frac

if st.button('Predict Probability'):
    if pipe is None:
        st.error("Model not loaded. Place 'pipe.pkl' next to this script and re-run.")
    else:
        # Basic sanity checks
        if target <= 0:
            st.error("Please enter a valid target (> 0).")
        else:
            total_balls_completed = overs_to_balls(overs)
            balls_left = 120 - total_balls_completed
            if balls_left <= 0:
                st.error("No balls left (overs indicate end of innings). Check the 'Overs completed' value.")
            else:
                runs_left = target - score
                wickets_left = 10 - int(wickets_out)

                # avoid divide by zero for C.R.R.
                crr = 0.0
                if overs > 0:
                    # compute C.R.R. as runs per over (score / overs_completed)
                    # if overs provided as decimal with balls, convert to overs fraction
                    overs_int = int(overs)
                    balls_frac = round((overs - overs_int) * 10)
                    overs_completed_fraction = overs_int + balls_frac / 6.0
                    if overs_completed_fraction > 0:
                        crr = score / overs_completed_fraction
                # required run rate
                rrr = (runs_left * 6) / balls_left

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

                # predict_proba expects same pre-processing as training
                try:
                    result = pipe.predict_proba(input_df)
                    loss = result[0][0]
                    win = result[0][1]
                    st.header(f"{batting_team} - {round(win*100, 2)}%")
                    st.header(f"{bowling_team} - {round(loss*100, 2)}%")
                except Exception as e:
                    st.error("Prediction failed. Make sure your model pipeline accepts the input features in the same format.")
                    st.exception(e)