
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import sklearn
print(f"Sklearn version: {sklearn.__version__}")

# Mock data
df = pd.DataFrame({
    'batting_team': ['A', 'B', 'A'],
    'bowling_team': ['B', 'A', 'B'],
    'city': ['X', 'Y', 'X'],
    'runs_left': [10, 20, 30]
})

print("Testing sparse=False...")
try:
    trf = ColumnTransformer([
        ('trf', OneHotEncoder(sparse=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
    ],
    remainder='passthrough')
    
    res = trf.fit_transform(df)
    print("Success with sparse=False")
except Exception as e:
    print(f"Error with sparse=False: {e}")

print("Testing sparse_output=False...")
try:
    trf = ColumnTransformer([
        ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
    ],
    remainder='passthrough')
    
    res = trf.fit_transform(df)
    print("Success with sparse_output=False")
except Exception as e:
    print(f"Error with sparse_output=False: {e}")
