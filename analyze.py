import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class Performer:
    num_label_0_train = 0
    num_label_1_train = 0
    num_label_0_test = 0
    num_label_1_test = 0

    def __init__(self):
        self.__df = pd.read_csv(
            "./data/1.csv",
            dtype={
                'COMPANY': str,
                'YEAR': str,
                'ID': str,
                'LOSS': int,
                'CHEAT': int
            })
        self.dims = list(self.__df.columns)
        print(0, self.__df[self.__df["CHEAT"] == 0].shape)
        print(1, self.__df[self.__df["CHEAT"] == 1].shape)

    @staticmethod
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

    def balanceDataset(self, multiple, blackList):
        """
        平衡两种数据集的数量
        @ multiple 未舞弊:舞弊
        @ blackList
        """
        df = self.df
        label_0_set = df[df["CHEAT"] == 0]
        label_1_set = df[df["CHEAT"] == 1]
        if blackList:
            label_0_set, label_1_set = self.removeCheatedCom(
                label_0_set, label_1_set)

        row_0_num = label_0_set.shape[0]
        row_1_num = label_1_set.shape[0]

        if row_0_num > row_1_num:
            frac_num = min(multiple * row_1_num, row_0_num) / row_0_num
            label_0_set = label_0_set.sample(frac=frac_num).reset_index(
                drop=True)
        else:
            frac_num = min(multiple * row_0_num, row_1_num) / row_1_num
            label_1_set = label_1_set.sample(frac=frac_num).reset_index(
                drop=True)
        # print("label_0_set", label_0_set.shape)
        # print("label_1_set", label_1_set.shape)
        # 拼接 打乱 去index
        return pd.concat([label_0_set,
                          label_1_set]).sample(frac=1).reset_index(drop=True)

    def handleData(self, blackList, selectedDims, train_ratio, multiple):
        self.df = self.__df.copy()[selectedDims +
                                   ["CHEAT"]].dropna().reset_index(drop=True)
        df = self.balanceDataset(multiple=multiple, blackList=blackList)

        train_test = int(train_ratio * (df.shape[0]))

        train = df.loc[:train_test - 1]
        self.x_train = train[selectedDims]
        self.y_train = train["CHEAT"]

        label_0_train = train[train["CHEAT"] == 0]
        self.num_label_0_train = label_0_train.shape[0]
        self.label_0_x_train = label_0_train[selectedDims]
        self.label_0_y_train = label_0_train["CHEAT"]

        label_1_train = train[train["CHEAT"] == 1]
        self.num_label_1_train = label_1_train.shape[1]
        self.label_1_x_train = label_1_train[selectedDims]
        self.label_1_y_train = label_1_train["CHEAT"]

        test = df.loc[train_test:]
        self.x_test = test[selectedDims]
        self.y_test = test["CHEAT"]

        label_0_test = test[test["CHEAT"] == 0]
        self.num_label_0_test = label_0_test.shape[0]
        self.label_0_x_test = label_0_test[selectedDims]
        self.label_0_y_test = label_0_test["CHEAT"]

        label_1_test = test[test["CHEAT"] == 1]
        self.num_label_1_test = label_1_test.shape[0]
        self.label_1_x_test = label_1_test[selectedDims]
        self.label_1_y_test = label_1_test["CHEAT"]

    def calcRandomForest(self, n_estimators):
        RF = {}
        try:
            clf = RandomForestClassifier(n_estimators=n_estimators)
            clf.fit(self.x_train, self.y_train)

            RF["label_0_score_test"] = clf.score(self.label_0_x_test,
                                                 self.label_0_y_test)
            RF["label_1_score_test"] = clf.score(self.label_1_x_test,
                                                 self.label_1_y_test)
            RF["score_test"] = clf.score(self.x_test, self.y_test)

            RF["label_0_score_train"] = clf.score(self.label_0_x_train,
                                                  self.label_0_y_train)
            RF["label_1_score_train"] = clf.score(self.label_1_x_train,
                                                  self.label_1_y_train)
            RF["score_train"] = clf.score(self.x_train, self.y_train)

            RF["feature_importances"] = clf.feature_importances_.tolist()
        except Exception as e:
            print("LogisticRegression")
            print(str(e))
            print(repr(e))
        finally:
            return RF

    def calcLogisticRegression(self, LR_type, kwargs):
        LR = {}
        try:
            clf = LogisticRegression()
            clf.fit(self.x_train, self.y_train)
            if LR_type == "2":
                coefficient = kwargs["data"]["coefficient"]
                intercept = kwargs["data"]["intercept"]
                # print(clf.coef_)
                # print(clf.intercept_)
                clf.coef_ = np.array([coefficient])
                clf.intercept_ = np.array([intercept])
                # print(clf.coef_)
                # print(clf.intercept_)
            LR["label_0_score_test"] = clf.score(self.label_0_x_test,
                                                 self.label_0_y_test)
            LR["label_1_score_test"] = clf.score(self.label_1_x_test,
                                                 self.label_1_y_test)
            LR["score_test"] = clf.score(self.x_test, self.y_test)
            LR["label_0_score_train"] = clf.score(self.label_0_x_train,
                                                  self.label_0_y_train)
            LR["label_1_score_train"] = clf.score(self.label_1_x_train,
                                                  self.label_1_y_train)
            LR["score_train"] = clf.score(self.x_train, self.y_train)
            LR["coef"] = clf.coef_.tolist()
            LR["intercept"] = clf.intercept_.tolist()
        except Exception as e:
            print("LogisticRegression")
            print(str(e))
            print(repr(e))
        finally:
            return LR

    # blackList=False,
    # selectedDims=["LOSS", "TATA1", "CHCS", "OTHREC"],
    # train_ratio=0.7,
    # multiple=1,
    # n_estimators=4
    def res(self, blackList, selectedDims, train_ratio, multiple, n_estimators,
            LR_type, **kwargs):

        self.handleData(blackList, selectedDims, train_ratio, multiple)

        return self.calcRandomForest(
            n_estimators), self.calcLogisticRegression(LR_type, kwargs)


performer = Performer()

# result = clf.predict(x_test)
# plt.figure()
# plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
# plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
# plt.title('score: %f' % score)
# plt.legend()
# plt.show()
