from sklearn.ensemble import RandomForestClassifier
import pandas as pd
df1 = pd.read_csv(
    "./data/1.csv",
    dtype={
        'COMPANY': str,
        'YEAR': str,
        'ID': str,
        'LOSS': int,
        'CHEAT': int
    })
print(df1.head())
print(df1.info())

