import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("crop_recommendation.csv")

X = data[['N','P','K','temperature','humidity','rainfall']]
y = data['label']

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("crop_model.pkl", "wb"))

print("✅ Model created!")