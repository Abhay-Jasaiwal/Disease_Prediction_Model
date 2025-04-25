import pandas as pd

data = pd.read_csv("improved_disease_dataset.csv")
print("📋 Columns in your dataset are:")
print(data.columns.tolist())
