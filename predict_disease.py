import pandas as pd
import joblib

model = joblib.load("disease_prediction_model.joblib")

data = pd.read_csv("improved_disease_dataset.csv")
symptoms = data.columns[:-1]  

print("\nPlease answer with 'yes' or 'no' for the following symptoms:\n")
user_input = []

for symptom in symptoms:
    answer = input(f"Do you have {symptom}? (yes/no): ").strip().lower()
    user_input.append(1 if answer == 'yes' else 0)

user_df = pd.DataFrame([user_input], columns=symptoms)

prediction = model.predict(user_df)[0]
print(f"\n The model predicts that you may have: **{prediction}**")
