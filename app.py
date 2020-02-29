from flask import Flask, request, render_template
import json
from analyze import performer

# blackList True or False   是否去掉曾经舞弊过的公司
# selectedDims  特征选择的维度
# train_ratio  训练集，测试集划分比例
# multiple  未舞弊:舞弊
# n_estimators  随机森林构成几棵树

app = Flask(__name__)

from flask_cors import CORS
# CORS(app, resources={r"/*": {"origins": "*"}})  # 允许所有域名跨域
CORS(
    app, resources={r"/predicting_financial_fraud/*": {
        "origins": "*"
    }})  # 只允许此接口所有域名跨域
CORS(
    app, resources={r"/search_information/*": {
        "origins": "*"
    }})  # 只允许此接口所有域名跨域


@app.route('/')
def index():
    return render_template('index.html')


# res(blackList=False,
#     selectedDims=["LOSS", "TATA1", "CHCS", "OTHREC"],
#     train_ratio=0.7,
#     multiple=1,
#     n_estimators=4)


@app.route('/predicting_financial_fraud', methods=['POST'])
def predicting_financial_fraud():
    try:
        data = json.loads(request.get_data().decode("utf-8"))
        if len(data["selectedDims"]) == 0:
            return "No Selected Dims !", 400
        RF, LR = performer.res(
            blackList=data["blackList"],
            selectedDims=data["selectedDims"],
            train_ratio=data["train_ratio"],
            multiple=data["multiple"],
            n_estimators=data["n_estimators"],
            LR_type=data["LR_type"],
            data=data)
        # print("RF", RF)
        # print("LR", LR)
        return {"RandomForest": RF, "LogisticRegression": LR}
    except Exception as e:
        print("predicting_financial_fraud")
        print(str(e))
        print(repr(e))
    # print("RF", RF)
    # print("LR", LR)
    # print("未舞弊公司正确预测率：\t", label_0_score)
    # print("舞弊公司正确预测率：\t", label_1_score)
    # print("总体正确预测率：\t", score)


@app.route('/search_information', methods=['POST'])
def search_information():
    try:
        data = json.loads(request.get_data().decode("utf-8"))
        if "info" not in data:
            return "", 400
        if data["info"] == "dataset":
            return {
                "label_0_train": performer.num_label_0_train,
                "label_1_train": performer.num_label_1_train,
                "label_0_test": performer.num_label_0_test,
                "label_1_test": performer.num_label_1_test
            }
        elif data["info"] == "dims":
            return {"dims": performer.dims}
    except:
        pass


if __name__ == '__main__':
    app.run()
