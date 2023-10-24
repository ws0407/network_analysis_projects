# from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QInputDialog, QMessageBox
import json

import onnx
import os
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from time import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

load_model_name = '/tmp/pycharm_project_880/models/best_ckpt_epoch14_testacc0.9965.onnx'
ROOT_DIR = '/tmp/pycharm_project_880/'
labels = ['JD', 'NCMusic', 'TED', 'Amazon', 'Baidu', 'Bing', 'Douban', 'Facebook',
                  'Google', 'IMDb', 'Instagram', 'iQIYI', 'QQMail', 'Reddit', 'Taobao',
                  'Tieba', 'Twitter', 'SinaWeibo', 'Youku', 'Youtube']

results = []

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
    img = img.crop((int(img.size[0] / 9), int(img.size[1] / 9), int(img.size[0] / 9 * 7), int(img.size[1] / 20 * 19)))
    img.save(ROOT_DIR + 'result/c_m.png')
    results.append({
        "type": "image",
        "title": "Confusion Matrix",
        "description": "Null",
        "data": "c_m.png"
    })


def test_ebsnn(sample_i):
    print('RUN EBSNN...')
    onnx_model = onnx.load(load_model_name)
    ort_session = ort.InferenceSession(load_model_name)
    st = time()

    one_sample = [
        np.load(ROOT_DIR + 'data/sample.npy'),
        np.load(ROOT_DIR + 'data/label.npy')
    ]
    x = one_sample[0]
    y = one_sample[1]
    outputs = ort_session.run(None, {'input': x})
    y_hat = np.argmax(outputs[0], axis=-1)
    c_m = confusion_matrix(y, y_hat)
    plot_confusion_matrix(c_m, labels)
    x_ = classification_report(y_true=y, y_pred=y_hat)
    x_ = x_.replace('macro avg', 'macro_avg')
    x_ = x_.replace('weighted avg', 'weighted_avg')
    for i in range(len(labels)):
        x_ = x_.replace(' ' + str(i) + '.0 ', labels[i], 1)
        x_ = x_.replace(' ' + str(i) + ' ', labels[i], 1)
    print(x_)
    header = ['Metrics'] + x_[:x_.find('\n\n')].split()
    body = x_[x_.find('\n\n') + 2:].split('\n')
    body = [i.split()[0:1] + [''] * (len(header) - len(i.split())) + i.split()[1:] for i in body]

    results.append({
        "type": "table",
        "title": "Classification Report",
        "data": {
            "thead": header,
            "tbody": body
        }
    })


# def on_button_clicked():
#     sample_i, okPressed = QInputDialog.getInt(window, "Test Sample", "Sample Number:",
#                                               np.random.randint(0, 100), 0, 100, 1)
#     if okPressed:
#         alert = QMessageBox()
#         res = test_ebsnn(sample_i)
#         alert.setText(res)
#         alert.exec_()


if __name__ == '__main__':
    # app = QApplication([])
    # window = QWidget()
    # layout = QVBoxLayout()
    # test_btn = QPushButton('Start')
    # layout.addWidget(test_btn)
    # test_btn.clicked.connect(on_button_clicked)
    # window.setLayout(layout)
    # window.show()
    # app.exec_()
    test_ebsnn(1)
    print(results)
    with open(ROOT_DIR + 'result/result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

    # with open(result_path + 'result.json', 'r', encoding='utf-8') as f1:
    #     res = json.load(f1)

    # print(res)

