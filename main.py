from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def balanceDataSet(df):
    label_0_set = df[df["CHEAT"] == 0]
    label_1_set = df[df["CHEAT"] == 1]
    row_0_num = label_0_set.shape[0]
    row_1_num = label_1_set.shape[0]
    multiple = 1.7  # 数据集0,1比例
    if row_0_num > row_1_num:
        frac_num = min(multiple * row_1_num, row_0_num) / row_0_num
        label_0_set = label_0_set.sample(frac=frac_num).reset_index(drop=True)
    else:
        frac_num = min(multiple * row_0_num, row_1_num) / row_1_num
        label_1_set = label_1_set.sample(frac=frac_num).reset_index(drop=True)
    print("label_0_set", label_0_set.shape)
    print("label_1_set", label_1_set.shape)
    return pd.concat([label_0_set, label_1_set])


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

df1 = balanceDataSet(df1).sample(frac=1).reset_index(drop=True)

train_test = int(0.7 * (df1.shape[0]))

train = df1.loc[:train_test - 1]
x_train = train[["LOSS", "TATA1" "CHCS", "OTHREC"]]
y_train = train["CHEAT"]

test = df1.loc[train_test:]
x_test = test[["LOSS", "TATA1", "CHCS", "OTHREC"]]
y_test = test["CHEAT"]

clf = RandomForestClassifier(n_estimators=4)
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
result = clf.predict(x_test)
plt.figure()
plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
plt.title('score: %f' % score)
plt.legend()
plt.show()
