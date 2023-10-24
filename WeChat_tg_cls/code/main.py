import json

from matplotlib import pyplot as plot
import numpy as np
from sklearn import linear_model
import joblib as jl
from ast import Num
from sys import executable
from numpy.core.fromnumeric import mean
from numpy.core.numeric import identity
from pandas.core.frame import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
from pandas import concat
import csv,os
from PIL import Image
import time
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import sklearn.naive_bayes as nb

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import *
# import tensorflow as tf
import operator
labels = ['文本', '图片', '朋友圈', '视频通话', '红包', '发送位置', '发送视频', '发送语音',
          'telegram文本', 'telegram图片', 'telegram语音', 'telegram视频', 'telegram发送文件',
          '语音通话', '读公众号', '转账', '打开小程序']
ROOT_DIR = '/tmp/pycharm_project_468/'

results = []


# 标签0：文本
# 标签1：图片
# 标签2：朋友圈
# 标签3：视频通话
# 标签4：红包
# 标签5：发送位置
# 标签6：发送视频
# 标签7：发送语音
# 标签8：telegram文本
# 标签9：telegram图片
# 标签10：telegram语音
# 标签11：telegram视频
# 标签12：telegram发送文件
# 标签13：语音通话
# 标签14：读公众号
# 标签15：转账
# 标签16：打开小程序


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(64, 48), dpi=100)
    np.set_printoptions(precision=2)
    cm = np.array(cm)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.0001:
            plt.text(x_val, y_val - 0.05, "%0.4f" % (c,), color='red', fontsize=30, va='center', ha='center')

    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.binary)
    # plt.title(title, fontdict={'size': 100})
    plt.colorbar()
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0:
            plt.text(x_val, y_val + 0.15, "(%d)" % (c,), color='c', fontsize=30, va='center', ha='center')
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.ylabel('Actual label', {'size': 60})
    plt.xlabel('Predict label', {'size': 60})

    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    # show confusion matrix
    plt.savefig(ROOT_DIR + 'result/c_m.png', format='png')
    # plt.show()
    plt.close()

    img = Image.open(ROOT_DIR + 'result/c_m.png')
    print(img.size)
    img = img.crop((int(img.size[0] / 7), int(img.size[1] / 9), int(img.size[0] / 7 * 6), int(img.size[1] / 9 * 8)))
    img.save(ROOT_DIR + 'result/c_m.png')
    results.append({
        "type": "image",
        "title": "Confusion Matrix",
        "description": "Null",
        "data": "c_m.png"
    })




# 构建训练数据，预处理数据

data_path = ROOT_DIR + 'data/' + os.listdir(ROOT_DIR + 'data/')[0]
featuresfinal = pd.read_csv(data_path, encoding='gbk')

featuresfinal=np.array(featuresfinal)
traffic_feature=[]
traffic_target=[]

for content in featuresfinal:
    content=list(content)
    traffic_feature.append(content[2:11])
    traffic_target.append(content[-1])
scaler = StandardScaler() # 标准化转换 
scaler.fit(traffic_feature)  # 训练标准化对象
traffic_feature= scaler.transform(traffic_feature)
# 分割数据集
feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target, test_size=0.5,random_state=0,stratify=traffic_target)


# 建立模型并且保存加载
model_RR=RandomForestClassifier()   
model_RR.fit(feature_train,target_train)

# 保存 model
jl.dump(model_RR, ROOT_DIR + 'model/model_RR.pkl')

print("模型保存成功")

# 加载 model
clf = jl.load(ROOT_DIR + 'model/model_RR.pkl')
#输入测试集
Y_pred = clf.predict(feature_test)
#输出标签类别
print(Y_pred)
# lst=['微信文本','微信图片','微信朋友圈','微信视频通话','微信发送红包',
# '微信发送位置','微信发送视频','微信发送语音','Telegram文本','Telegram图片',
# 'Telegram语音','Telegram视频','Telegram发送文件','微信语音通话','微信读公众号',
# '微信转账','微信打开小程序']
#================随机森林分类器的预测结果======================
print("随机森林结果如下：\n")
print(accuracy_score(Y_pred, target_test))
print("_______________________________________")
conf_mat = confusion_matrix(target_test, Y_pred)
plot_confusion_matrix(conf_mat, [i for i in range(len(labels))])
print(conf_mat)
print("_______________________________________")
x_ = classification_report(target_test, Y_pred, digits=7)
x_ = x_.replace('macro avg', 'macro_avg')
x_ = x_.replace('weighted avg', 'weighted_avg')
for i in range(len(labels)):
    x_ = x_.replace(str(i)+'.0', labels[i], 1)
print(x_)
header = ['Metrics'] + x_[:x_.find('\n\n')].split()
body = x_[x_.find('\n\n')+2:].split('\n')
body = [i.split()[0:1] + [''] * (len(header) - len(i.split())) + i.split()[1:] for i in body]

results.append({
    "type": "table",
    "title": "Classification Report",
    "data": {
        "thead": header,
        "tbody": body
    }
})

print("_______________________________________")


print(results)
with open(ROOT_DIR + 'result/result.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False)

# with open(result_path + 'result.json', 'r', encoding='utf-8') as f1:
#     res = json.load(f1)

# print(res)


