import pandas as pd

df = pd.read_csv("./data/assignTTSWING.csv")
print(df.columns)
print(df.describe())
print(df.isnull().sum())
