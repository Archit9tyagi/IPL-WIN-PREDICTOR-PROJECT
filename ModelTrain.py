#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd


# In[21]:


match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')


# In[23]:


match.head()


# In[25]:


match.shape

# Find total_runs scored in Both the Innings
# In[27]:


total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()


# In[28]:


delivery.head()

# Find total_runs scored in First Innings 
# In[31]:


total_score_df = total_score_df[total_score_df['inning'] == 1]


# In[33]:


total_score_df

# MERGE THE DATAFRAME OF TOTAL RUNS WITH MATCH DATAFRAME
# In[35]:


match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')


# In[37]:


print(match_df.columns.tolist())


# In[ ]:




# DATA PREPROCESSING 
# REMOVE THOSE TEAMS WHO LEFT THE IPL AND REPLACE TWO TEAMS WHICH CHANGE THEIR NAME.
# In[40]:


match_df['team1'].unique()


# In[42]:


teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]


# In[44]:


match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# In[46]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]


# In[48]:


match_df.shape


# In[50]:


match_df = match_df[match_df['dl_applied'] == 0]


# In[52]:


match_df = match_df[['match_id','city','winner','total_runs']]


# In[54]:


delivery_df = match_df.merge(delivery,on='match_id')


# In[56]:


delivery_df = delivery_df[delivery_df['inning'] == 2]


# In[58]:


delivery_df


# In[60]:


# Convert to numeric (ignore non-numeric values)
# delivery_df['total_runs_y'] = pd.to_numeric(delivery_df['total_runs_y'], errors='coerce')

# Now compute the cumulative sum grouped by match_id
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()


# In[62]:


delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']


# In[64]:


delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])


# In[66]:


delivery_df


# In[68]:


# Step 1: Replace non-numeric or missing values properly
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].replace(['', 'NaN', None, np.nan], '0')

# Step 2: Convert to numeric safely
delivery_df['player_dismissed'] = pd.to_numeric(delivery_df['player_dismissed'], errors='coerce').fillna(0).astype(int)

# Step 3: Convert 0/1 logic properly
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: 1 if x != 0 else 0)

# Step 4: Now cumsum works fine
delivery_df['wickets_fallen'] = delivery_df.groupby('match_id')['player_dismissed'].cumsum()

# Step 5: Calculate remaining wickets
delivery_df['wickets'] = 10 - delivery_df['wickets_fallen']

# Step 6: View result
delivery_df.tail()


# In[70]:


delivery_df.head(10)


# In[72]:


# crr = runs/overs
delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])


# In[74]:


delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']


# In[76]:


def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0


# In[78]:


delivery_df['result'] = delivery_df.apply(result,axis=1)


# In[79]:


final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]


# In[82]:


final_df = final_df.sample(final_df.shape[0])


# In[84]:


final_df.sample()


# In[86]:


final_df.dropna(inplace=True)


# In[88]:


final_df = final_df[final_df['balls_left'] != 0]


# In[90]:


X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[102]:


X_train


# In[104]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
],
remainder='passthrough')


# In[106]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[108]:


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline(steps=[
    ('step1', trf),                      # your transformer (ColumnTransformer or preprocessing)
    ('step2', RandomForestClassifier())  # model
])


# In[110]:


pipe.fit(X_train,y_train)


# In[111]:


y_pred = pipe.predict(X_test)


# In[112]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[44]:


pipe.predict_proba(X_test)[50]


# In[45]:


def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))
    


# In[46]:


def match_progression(x_df, match_id, pipe):
    # 1. Filter match
    match = x_df[x_df['match_id'] == match_id]
    
    # 2. Keep only end-of-over rows (ball == 6)
    match = match[match['ball'] == 6]

    # 3. Select only required model features
    temp_df = match[['batting_team','bowling_team','city',
                     'runs_left','balls_left','wickets',
                     'total_runs_x','crr','rrr']].copy()
    
    # Remove missing data & rows where match is already finished
    temp_df = temp_df.dropna()
    temp_df = temp_df[temp_df['balls_left'] > 0]

    # 4. Predict win-loss probabilities
    result = pipe.predict_proba(temp_df)

    temp_df['lose'] = np.round(result[:, 0] * 100, 1)   # class 0
    temp_df['win']  = np.round(result[:, 1] * 100, 1)   # class 1

    # 5. Define over number
    temp_df['end_of_over'] = np.arange(1, len(temp_df) + 1)

    # 6. RUNS SCORED IN EACH OVER
    target = temp_df['total_runs_x'].iloc[0]               # target score
    runs_left = temp_df['runs_left'].values                # runs left after each over

    runs_before = np.insert(runs_left, 0, target)          # prepend target
    runs_after = runs_left                                 # after overs

    temp_df['runs_after_over'] = runs_before[:-1] - runs_after

    # 7. WICKETS LOST IN EACH OVER
    wkts_after = temp_df['wickets'].values                 # wickets left after each over
    wkts_before = np.insert(wkts_after, 0, 10)             # start with 10 wickets

    temp_df['wickets_in_over'] = wkts_before[:-1] - wkts_after

    # 8. Final cleaned DataFrame
    temp_df = temp_df[['end_of_over',
                       'runs_after_over',
                       'wickets_in_over',
                       'lose',
                       'win']]

    print("Target =", target)
    return temp_df, target


# In[47]:


temp_df,target = match_progression(delivery_df,74,pipe)
temp_df


# In[48]:


import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))


# In[49]:


teams


# In[50]:


delivery_df['city'].unique()


# In[51]:


import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))

