import pickle
import sklearn
print("Current sklearn version:", sklearn.__version__)

with open("pipe.pkl","rb") as f:
    obj = pickle.load(f)
print(obj)