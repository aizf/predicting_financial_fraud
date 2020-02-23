from analyze import res

# blackList True or False   是否去掉曾经舞弊过的公司
# selectedDims  特征选择的维度
# train_test_ratio  训练集，测试集划分比例
# multiple  未舞弊:舞弊
# n_estimators  随机森林构成几棵树
res(blackList=False,
    selectedDims=["LOSS", "TATA1", "CHCS", "OTHREC"],
    train_test_ratio=0.7,
    multiple=1,
    n_estimators=4)
