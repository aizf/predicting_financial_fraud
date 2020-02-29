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
        self.__otherDims = ["COMPANY", "DATE", "YEAR", "ID", "CHEAT"]
        self.dims = [
            dim for dim in self.__df.columns if dim not in self.__otherDims
        ]
        list(
            set(self.__df.columns).difference(set(self.__otherDims))
            )
        self.headerDims = ["COMPANY", "CHEAT"]
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

    @staticmethod
    def balanceDataset(label_0_set, label_1_set, multiple):
        """
        平衡两种数据集的数量
        @ multiple 未舞弊:舞弊
        """
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
        # print("label_1_set", label_1_set)
        # 拼接 打乱 去index
        return label_0_set, label_1_set

    def handleData(self, blackList, selectedDims, train_ratio, multiple):
        # copy
        self.df = self.__df.copy()[selectedDims +
                                   self.headerDims].dropna().reset_index(
                                       drop=True)
        label_0_set = self.df[self.df["CHEAT"] == 0]
        label_1_set = self.df[self.df["CHEAT"] == 1]
        if blackList:
            label_0_set, label_1_set = self.removeCheatedCom(
                label_0_set, label_1_set)

        label_0_set, label_1_set = self.balanceDataset(
            label_0_set, label_1_set, multiple=multiple)
        # print("label_0_set")
        # print(label_0_set)
        # print("label_1_set")
        # print(label_1_set)

        train_get_num_0 = int(train_ratio * (label_0_set.shape[0]))
        train_get_num_1 = int(train_ratio * (label_1_set.shape[0]))

        label_0_train = label_0_set.iloc[:train_get_num_0]
        self.num_label_0_train = label_0_train.shape[0]
        self.label_0_x_train = label_0_train[selectedDims]
        self.label_0_y_train = label_0_train["CHEAT"]

        label_1_train = label_1_set.iloc[:train_get_num_1]
        self.num_label_1_train = label_1_train.shape[0]
        self.label_1_x_train = label_1_train[selectedDims]
        self.label_1_y_train = label_1_train["CHEAT"]

        train = pd.concat(
            [label_0_train,
             label_1_train]).sample(frac=1).reset_index(drop=True)
        self.x_train = train[selectedDims]
        self.y_train = train["CHEAT"]

        label_0_test = label_0_set.iloc[train_get_num_0:]
        self.num_label_0_test = label_0_test.shape[0]
        self.label_0_x_test = label_0_test[selectedDims]
        self.label_0_y_test = label_0_test["CHEAT"]

        label_1_test = label_1_set.iloc[train_get_num_1:]
        self.num_label_1_test = label_1_test.shape[0]
        self.label_1_x_test = label_1_test[selectedDims]
        self.label_1_y_test = label_1_test["CHEAT"]

        test = pd.concat([label_0_test,
                          label_1_test]).sample(frac=1).reset_index(drop=True)
        self.x_test = test[selectedDims]
        self.y_test = test["CHEAT"]

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
