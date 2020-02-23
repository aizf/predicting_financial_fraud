from flask import Flask, request
from flask_cors import CORS
import json
from analyze import res

# blackList True or False   是否去掉曾经舞弊过的公司
# selectedDims  特征选择的维度
# train_test_ratio  训练集，测试集划分比例
# multiple  未舞弊:舞弊
# n_estimators  随机森林构成几棵树

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 允许所有域名跨域


@app.route('/')
def index():
    return 'Index Page'


# res(blackList=False,
#     selectedDims=["LOSS", "TATA1", "CHCS", "OTHREC"],
#     train_test_ratio=0.7,
#     multiple=1,
#     n_estimators=4)


@app.route('/predicting_financial_fraud', methods=['POST'])
def predicting_financial_fraud():
    try:
        data = json.loads(request.get_data().decode("utf-8"))
        label_0_score, label_1_score, score = res(
            blackList=data["blackList"],
            selectedDims=data["selectedDims"],
            train_test_ratio=data["train_test_ratio"],
            multiple=data["multiple"],
            n_estimators=data["n_estimators"])
        return {
            "label_0_score": label_0_score,
            "label_1_score": label_1_score,
            "score": score
        }
    except:
        pass
    # print("未舞弊公司正确预测率：\t", label_0_score)
    # print("舞弊公司正确预测率：\t", label_1_score)
    # print("总体正确预测率：\t", score)
