import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def removeCheatedCom(label_0_set, label_1_set):
    """
    去掉曾经舞弊过的公司
    """
    cheatedComs = []
    for index, row in label_1_set.iterrows():
        cheatedComs.append(row["COMPANY"])
    cheatedComs = list(set(cheatedComs))
    cheatedComs.sort()
    # print(cheatedComs)
    for com in cheatedComs:
        label_0_set = label_0_set[label_0_set["COMPANY"] != com]
    return label_0_set, label_1_set


def balanceDataSet(df, multiple=1, blackList=False):
    """
    平衡两种数据集的数量
    @ multiple 未舞弊:舞弊
    @ blackList
    """
    label_0_set = df[df["CHEAT"] == 0]
    label_1_set = df[df["CHEAT"] == 1]
    if blackList:
        label_0_set, label_1_set = removeCheatedCom(label_0_set, label_1_set)

    row_0_num = label_0_set.shape[0]
    row_1_num = label_1_set.shape[0]

    if row_0_num > row_1_num:
        frac_num = min(multiple * row_1_num, row_0_num) / row_0_num
        label_0_set = label_0_set.sample(frac=frac_num).reset_index(drop=True)
    else:
        frac_num = min(multiple * row_0_num, row_1_num) / row_1_num
        label_1_set = label_1_set.sample(frac=frac_num).reset_index(drop=True)
    # print("label_0_set", label_0_set.shape)
    # print("label_1_set", label_1_set.shape)
    # 拼接 打乱 去index
    return pd.concat([label_0_set,
                      label_1_set]).sample(frac=1).reset_index(drop=True)


def res(blackList=False,
        selectedDims=["LOSS", "TATA1", "CHCS", "OTHREC"],
        train_test_ratio=0.7,
        multiple=1,
        n_estimators=4):
    df1 = pd.read_csv(
        "./data/1.csv",
        dtype={
            'COMPANY': str,
            'YEAR': str,
            'ID': str,
            'LOSS': int,
            'CHEAT': int
        })
    # print(df1.head())
    # print(df1.info())
    df1 = balanceDataSet(df1, multiple=multiple, blackList=blackList)

    train_test = int(train_test_ratio * (df1.shape[0]))

    train = df1.loc[:train_test - 1]
    x_train = train[selectedDims]
    y_train = train["CHEAT"]

    test = df1.loc[train_test:]
    x_test = test[selectedDims]
    y_test = test["CHEAT"]

    label_0_test = test[test["CHEAT"] == 0]
    label_0_x_test = label_0_test[selectedDims]
    label_0_y_test = label_0_test["CHEAT"]

    label_1_test = test[test["CHEAT"] == 1]
    label_1_x_test = label_1_test[selectedDims]
    label_1_y_test = label_1_test["CHEAT"]

    # RandomForest
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(x_train, y_train)
    RF = {}
    RF["label_0_score"] = clf.score(label_0_x_test, label_0_y_test)
    RF["label_1_score"] = clf.score(label_1_x_test, label_1_y_test)
    RF["score"] = clf.score(x_test, y_test)
    RF["feature_importances"] = clf.feature_importances_.tolist()
    # print("\n", "RandomForest")
    # print(clf.feature_importances_)
    # print(clf.oob_score_)
    # print(clf.get_params())
    # print("\n")

    # LogisticRegression
    clf1 = LogisticRegression()
    clf1.fit(x_train, y_train)
    LR = {}
    LR["label_0_score"] = clf1.score(label_0_x_test, label_0_y_test)
    LR["label_1_score"] = clf1.score(label_1_x_test, label_1_y_test)
    LR["score"] = clf1.score(x_test, y_test)
    LR["coef"] = clf1.coef_.tolist()
    LR["intercept"] = clf1.intercept_.tolist()
    # print("\n", "LogisticRegression")
    # print(clf1.coef_)
    # print("\n")

    return RF, LR


# result = clf.predict(x_test)
# plt.figure()
# plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
# plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
# plt.title('score: %f' % score)
# plt.legend()
# plt.show()
