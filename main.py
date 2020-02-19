from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

df1 = df1.sample(frac=1).reset_index()

train_test = int(0.7 * (df1.shape[0]))

train = df1.loc[:train_test-1]
x_train = train[["LOSS", "TATA1", "CHCS", "OTHREC"]]
y_train = train["CHEAT"]

test = df1.loc[train_test:]
x_test = test[["LOSS", "TATA1", "CHCS", "OTHREC"]]
y_test = test["CHEAT"]

clf = RandomForestClassifier(n_estimators=10)
clf.fit(x_train,y_train)
score = clf.score(x_test, y_test)
result = clf.predict(x_test)
plt.figure()
plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
plt.title('score: %f'%score)
plt.legend()
plt.show()
