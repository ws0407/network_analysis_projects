# from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import os
import sys
import dpkt
import numpy

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # using specific GPU
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

from compatible.likelihoods import Gaussian
from compatible.kernels import RBF, White
from gpflow.models.svgp import SVGP
# from gpflow.training import AdamOptimizer, ScipyOptimizer
from scipy.stats import mode
from scipy.cluster.vq import kmeans2
import gpflow
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow import autoflow, params_as_tensors, ParamList
import pandas as pd
import itertools

pd.options.display.max_rows = 999
import gpflow_monitor

from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.io import loadmat
from gpflow_monitor import *

print('tf_ver:', tf.__version__, 'gpflow_ver:', gpflow.__version__)
from tensorflow.python.client import device_lib

print('avail devices:\n' + '\n'.join([x.name for x in device_lib.list_local_devices()]))
from jack_utils.common import time_it
import sys
import gpflow.training.monitor as mon

# our impl
from dgp_graph import *

# %%

plt.rcParams['figure.dpi'] = 100
np.random.seed(123456)

# %% md

# Load data

# %% md

# Input
# filename

# %%
file_path = '/tmp/pycharm_project_992/data/'
result_path = '/tmp/pycharm_project_992/result/'
results = []

filename = os.path.join(file_path, os.listdir(file_path)[0])
print(filename)
time.sleep(100)

# filename = './data/facebook_chat_4b.pcap'

# %%

counter = 0
last_time = 0
data_list = []

for ts, pkt in dpkt.pcap.Reader(open(filename, 'rb')):
    if last_time == 0:
        last_time = int(ts)
    else:
        cur_time = int(ts)
        if cur_time == last_time:
            counter += 1
        else:
            data_list.append(counter)
            last_time = cur_time
            counter = 1
data_list.append(counter)

# %%

# plt.plot(data_list)
# plt.savefig(result_path + '1.png')
# plt.close()

results.append({
    "type": "text",
    "title": "1. 数据集可视化",
    "data": "横坐标表示流量的Time Step，纵坐标表示流量的Expression Level。"
})

results.append({
    "type": "line chart",
    "title": "Show Dataset",
    "description": "Null",
    "labels": [i for i in range(len(data_list))],
    "data": [
        {
          "name": "",
          "data": data_list
        }
    ]
})

# %% md

# Build data

# %%

data_list = np.asarray(data_list)
l = len(data_list)
train_size = int(0.6 * l)
test_size = l - train_size

# %%
M = l//3
win = 5
X_train = np.zeros((train_size - win, win))
Y_train = np.zeros((train_size - win, 1))
X_test = np.zeros((test_size - win, win))
Y_test = np.zeros((test_size - win, 1))

# %%

for i in range(train_size - win):
    X_train[i] = data_list[i:i + win]
    Y_train[i, 0] = data_list[i + win]

# %%

for i in range(test_size - win):
    X_test[i] = data_list[train_size + i:train_size + i + win]
    Y_test[i, 0] = data_list[train_size + i + win]


# %%


# %%

def normalize_data(data, mu, std):
    res = (data - mu) / std
    return res


def unnormalize_data(data, mu, std):
    res = data * std + mu
    return res


# %%

trX0 = X_train
trY0 = Y_train

teX0 = X_test
teY0 = Y_test

# %%

trX0, teX0 = trX0[:, None, :], teX0[:, None, :]
trY0, teY0 = trY0[:, None, :], teY0[:, None, :]

# %%

mu_trX0, std_trX0 = np.mean(trX0, axis=0), np.std(trX0, axis=0)
mu_trY0, std_trY0 = np.mean(trY0, axis=0), np.std(trY0, axis=0)
# mu_teX0, std_teX0 = np.mean(teX0, axis=0), np.std(teX0, axis=0)
# mu_teY0, std_teY0 = np.mean(teY0, axis=0), np.std(teY0, axis=0)

# %%

trX = normalize_data(trX0, mu_trX0, std_trX0)
trY = normalize_data(trY0, mu_trY0, std_trY0)
teX = normalize_data(teX0, mu_trX0, std_trX0)
teY = normalize_data(teY0, mu_trY0, std_trY0)

# %%

gmat = np.eye(1)


# %% md

# Data visualization

# %%

def imshow2d(x, idx):
    plt.imshow(x.reshape(x.shape[0], -1))
    plt.colorbar()
    plt.savefig(result_path + str(idx) + '.png')
    plt.close()


# %%

imshow2d(trX0, 2)

# %%

imshow2d(trY0, 3)

# %%

imshow2d(teX, 4)

# %%

imshow2d(teY, 5)

results.append({
    "type": "text",
    "title": "2. 训练和测试数据的可视化",
    "data": ""
})

results.append({
    "type": "image",
    "title": "(1) Training X Visualization",
    "description": "Null",
    "data": "2.png"
})
results.append({
    "type": "image",
    "title": "(2) Training y Visualization",
    "description": "Null",
    "data": "3.png"
})
results.append({
    "type": "image",
    "title": "(3) Test X Visualization",
    "description": "Null",
    "data": "4.png"
})
results.append({
    "type": "image",
    "title": "(4) Test y Visualization",
    "description": "Null",
    "data": "4.png"
})


# %% md

# Training

# %%

nodes = gmat.shape[0]
# M = 5

Z = np.stack([kmeans2(trX[:, i], M, minit='points')[0] for i in range(nodes)], axis=1)  # (M=s2=10, n, d_in=5)
print('inducing points Z: {}'.format(Z.shape))

# %%

adj = gmat.astype('float64')
input_adj = adj  # adj  / np.identity(adj.shape[0]) /  np.ones_like(adj)

# %%

with gpflow.defer_build():
    m_dgpg = DGPG(trX, trY, Z, [win], Gaussian(), input_adj,
                  agg_op_name='concat3d', ARD=True,
                  is_Z_forward=True, mean_trainable=False, out_mf0=True,
                  num_samples=20, minibatch_size=80,
                  kern_type='RBF'
                  # kern_type='Matern32'
                  )
    # m_sgp = SVGP(X, Y, kernels, Gaussian(), Z=Z, minibatch_size=minibatch_size, whiten=False)
m_dgpg.compile()
model = m_dgpg

# %%

session = m_dgpg.enquire_session()
optimiser = gpflow.train.AdamOptimizer(0.01)
global_step = mon.create_global_step(session)

# %%

maxiter = 2000

exp_path = "./exp/tmp"
# exp_path="./exp/temp"


print_task = mon.PrintTimingsTask()\
    .with_name('print')\
    .with_condition(mon.PeriodicIterationCondition(10))
checkpoint_task = mon.CheckpointTask(checkpoint_dir=exp_path)\
    .with_name('checkpoint')\
    .with_condition(mon.PeriodicIterationCondition(15))
with mon.LogdirWriter(exp_path) as writer:
    tensorboard_task = mon.ModelToTensorBoardTask(writer, model)\
        .with_name('tensorboard')\
        .with_condition(mon.PeriodicIterationCondition(100))\
        .with_exit_condition(True)
    monitor_tasks = [tensorboard_task, checkpoint_task]  # [print_task, tensorboard_task]

    with mon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
        optimiser.minimize(model, step_callback=monitor, global_step=global_step, maxiter=maxiter)

# %% md

# Results

# %%

from jack_utils.my_metrics import *
import matplotlib.pyplot as plt
# % matplotlib
# inline


# %%

def assess_model_rmse(model, X_batch, Y_batch, S=10):
    m, v = model.predict_y(X_batch, S)
    pred = np.mean(m, axis=0)
    var = np.mean(v, axis=0)
    loss = np.sum((Y_batch.flatten() - pred.flatten()) ** 2)
    return loss, pred, var


def batch_assess_rmse(model, X, Y, batch_size=1, S=10):
    n_batches = max(int(len(X) / batch_size), 1)
    rms = len(X) - n_batches * batch_size
    losses, preds, varis = [], [], []
    Xr, Yr = X[-rms:, :], Y[-rms:, :]
    for X_batch, Y_batch in zip(np.split(X[:n_batches * batch_size], n_batches),
                                np.split(Y[:n_batches * batch_size], n_batches)):
        l, pred, vari = assess_model_rmse(model, X_batch, Y_batch, S=S)
        losses.append(l)
        preds.append(pred)
        varis.append(vari)
    if rms > 0:
        l, pred, vari = assess_model_rmse(model, Xr, Yr, S=S)
        losses.append(l)
        preds.append(pred)
        varis.append(vari)
    ndata = Y.shape[0] * Y.shape[1]
    avg_loss = np.sqrt(np.sum(losses) / ndata)
    y_pred = np.concatenate(preds)
    y_var = np.concatenate(varis)
    return avg_loss, y_pred, y_var


# %%

# make predictions of our model
sample_X0, sample_Y0 = teX0, teY0.squeeze()
sample_X, sample_Y = teX, teY.squeeze()

# %%

# % % time
pred_rmse, predY, varY = batch_assess_rmse(model, sample_X.reshape(sample_X.shape[0], -1),
                                           sample_Y.reshape(sample_Y.shape[0], -1))
print(pred_rmse)

# %%

predY0 = unnormalize_data(predY[:, :, None], mu_trY0, std_trY0).squeeze()

# %%

# show rmse of all baselines
metrics = [np_mae, np_rmse, np_mape]
e_dgp = [np.round(f(predY0, sample_Y0.squeeze()), 3) for f in metrics]
e_last = [np.round(f(sample_X0[:, :, -1], sample_Y0.squeeze()), 3) for f in metrics]
e_ha = [np.round(f(sample_X0.mean(axis=-1), sample_Y0.squeeze()), 3) for f in metrics]
e_mid = [np.round(f(np.median(sample_X0, axis=-1), sample_Y0.squeeze()), 3) for f in metrics]
print('metrics:\t[mae | rmse | mape]')
print('ours DGP:\t', e_dgp)
print('Last baseline:\t', e_last)
print('5s mean baseline:\t', e_ha)
print('5s median baseline:\t', e_mid)

# %%

pred = predY0
var = (varY * std_trY0).squeeze()

data = data_list[train_size + win:]

ks = 2
yli = 0.05

plt.plot(range(test_size - win), pred, label='Prediction')
plt.plot(range(test_size - win), data, label='Ground truth')
plt.fill_between(range(test_size - win), pred - ks * np.sqrt(var),
                 pred + ks * np.sqrt(var),
                 alpha=0.3)
# plt.ylim([ymin-yli, ymax+yli])
plt.legend()
plt.xlabel('Time step')
plt.ylabel('Expression level')
plt.show()
plt.savefig(result_path + '6.png')
plt.close()

results.append({
    "type": "text",
    "title": "3. 选择查看对应流量可视化折线图",
    "data": ""
})

results.append({
    "type": "image",
    "title": "Expression level - Time step Figure",
    "description": "流量可视化折线图",
    "data": "6.png"
})

results.append({
    "type": "table",
    "title": "4. 选择查看对应的流量预测结果",
    "data": {
        "thead": ["metrics", "mae", "rmse", "mape"],
        "tbody": [
            ["ours DGP:"] + e_dgp,
            ["Last baseline:"] + e_last,
            ["5s mean baseline:"] + e_ha,
            ["5s median baseline:"] + e_mid
        ]
    }
})

# results.append({
#     "type": "line chart",
#     "title": "Expression level - Time step Figure",
#     "description": "Null",
#     "labels": [i for i in range(test_size - win)],
#     "yTitle": "Expression level",
#     'xTitle': "Time step",
#     "data": [
#         {
#             "name": "Prediction",
#             "data": pred.tolist()
#         },
#         {
#             "name": "Ground truth",
#             "data": data.tolist()
#         },
#     ]
# })

# %% md

# Variance analysis

# %%

std_varY0 = np.sqrt(varY) * std_trY0.reshape(1, varY.shape[1])


# %%

def correct_rate(predY0, std_varY0, sample_Y0, ndev):
    predY0_ub = predY0 + std_varY0 * ndev
    predY0_lb = predY0 - std_varY0 * ndev
    tf_mat = np.logical_and(predY0_lb <= sample_Y0, sample_Y0 <= predY0_ub)
    correct_rate = np.sum(tf_mat) / np.product(tf_mat.shape)
    print('ndev=' + str(ndev), correct_rate)
    return correct_rate


# %%

correct_rate(predY0, std_varY0, sample_Y0, ndev=1)

# %%

correct_rate(predY0, std_varY0, sample_Y0, ndev=2)

# %%

correct_rate(predY0, std_varY0, sample_Y0, ndev=3)

# %%

import pandas as pd

z=model.layers[0].feature.Z.value.squeeze()

print('挖掘特征个数：%d\n'%M)
tmp = pd.DataFrame(z)   # .head(10)
tmp = numpy.array(tmp).tolist()
# print(tmp)

results.append({
    "type": "text",
    "title": "5. 选择查看所挖掘特征的表格展示",
    "data": "共挖掘到138个特征，每个特征为5维实数向量，下表每页展示10个特征，下图为所挖掘特征的t-SNE展示"
})

results.append({
    "type": "table",
    "title": "(1) 所挖掘特征的表格展示",
    "data": {
        "thead": [1, 2, 3, 4, 5],
        "tbody": tmp
    }
})

# ####

from sklearn.manifold import TSNE
z2d = TSNE(n_components=2, learning_rate='auto',
                  init='random').fit_transform(z)

plt.scatter(z2d[:,0],z2d[:,1])
plt.savefig(result_path + 't-SNE.png')
plt.close()

results.append({
    "type": "image",
    "title": "(2) 所挖掘特征的t-SNE展示",
    "description": "Null",
    "data": "t-SNE.png"
})


# ####





print(results)
with open(result_path + 'result.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False)

# with open(result_path + 'result.json', 'r', encoding='utf-8') as f1:
#     res = json.load(f1)

# print(res)
