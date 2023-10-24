# Author: Jiahui
# Project: Iqiyi local popularity prediction
# Time: 2016-10-16 ~ 10.30
# First edit: 2017.10.16
# Lastest vision: 2017.11.27

'''
    This file is used to preprocess for LSTM idea -- hierarchical sequence prediction

    # Input file: filtered_iqiyi

    # Method: Adaboost Decision Tree Regressor
    # Cross validation: train:test = 80%:20%

    # Output:

'''
import json
import sys
import gc, os
import gridnumcompute as gdn
import numpy as np
import sklearn
from sklearn.utils import shuffle
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow import keras
# from keras.layers.core import *
from tensorflow.keras.callbacks import EarlyStopping
import time
# import cPickle as pickle
import _pickle as pickle
from tensorflow.keras import optimizers
from math import log10
# import sPickle
import tensorflow as tf

def preprocess():
    '''
        read file of top 2000 viewed videos
    '''
    vdic = {}

    f = open('top2000','r')
    fv = f.readlines()
    f.close()

    for i in range(len(fv)):
        items = fv[i].split()
        vdic[items[1]] = i
    print('--- vdic DONE ---')


    '''
        Select sessions of top 1000 viewed videos

        Filters:
            -1) have been published before 2015-05-11
    '''

    f = open('../filtered_iqiyi', 'r')
    # f = open('head', 'r')
    fin = f.readlines()
    f.close()

    v_day_count = [set() for i in range(2000)]
    session_temp = []
    for i in range(len(fin)):
        items = fin[i].split()
        try:
            v_day_count[vdic[items[10]]].add(items[6])
            session_temp.append(fin[i])
        except:
            continue
    del fin
    gc.collect()
    print(len(session_temp))
    print('--- vdic day count DONE ---')

    indx = []
    for i in range(2000):
        if len(v_day_count[i])>= 13:
            indx.append(i)
    print('--- indx DONE ---')
    print(len(indx))

    f = open('sess_top2000', 'w')
    for i in range(len(session_temp)):
        items = session_temp[i].split()
        try:
            if vdic[items[10]] in indx:
            # session_temp.append(fin[i])
                f.write(items[5] + ' ' + items[6] + ' ' + items[7] + ' ' + items[8] + ' ' + items[9] + ' ' + items[10] + '\n')
        except:
            continue
    f.close()
    print('--- write DONE ---')

def new1000():
    f = open('new_top_2000', 'r')
    fv = f.readlines()
    f.close()

    vdic = {}
    for i in range(1000):
        items = fv[i].split()
        vdic[items[1]] = i

    f = open('sess_top2000', 'r')
    fin = f.readlines()
    f.close()

    f = open('sess_new1000', 'w')
    for i in range(len(fin)):
        try:
            if vdic[fin[i].split()[5]] >= 0:
                f.write(fin[i])
        except:
            continue
    f.close()

def hiersessfile():
    f = open('sess_new1000', 'r')
    fin = f.readlines()
    f.close()

    f = open('hier_sess_1000', 'w')
    for i in range(len(fin)):
        items = fin[i].split()
        n0, n1, n2, n3, n4 = gdn.gridnum(5, float(items[3]), float(items[4]))
        if n0 >= 0 and n0 < 4:
            f.write(str(n0) + ' ' + str(n1) + ' ' + str(n2) + ' ' + str(n3) + ' ' + str(n4) + ' ' + fin[i])
        else:
            continue
    f.close()

    f = open('non_hier_sess_1000', 'w')
    for i in range(len(fin)):
        items = fin[i].split()
        n0, n1, n2, n3, n4 = gdn.non_hier_gridnum(5, float(items[3]), float(items[4]))
        if n0 >= 0 and n0 < 4:
            f.write(str(n0) + ' ' + str(n1) + ' ' + str(n2) + ' ' + str(n3) + ' ' + str(n4) + ' ' + fin[i])
        else:
            continue
    f.close()

def sparsity(lev):
    f = open('new_top_2000', 'r')
    fv = f.readlines()
    f.close()

    vdic = {}
    for i in range(1000):
        items = fv[i].split()
        vdic[items[1]] = i

    f = open('non_hier_sess_1000', 'r')
    fin = f.readlines()
    f.close()

    reqarr = np.zeros((336-24, 1000, pow(4,lev+1)))
    # reqarr = [[[0 for k in range(4)] for i in range(1000)] for j in range(336-24)]
    for i in range(len(fin)):
        items = fin[i].split()

        a = time.strptime(items[6], "%Y-%m-%d")
        b = time.strptime(items[7], "%H:%M:%S")
        if a.tm_mday == 10:
            continue
        elif a.tm_mday > 10:
            tid = (a.tm_mday - 4 - 1)*24 + b.tm_hour
        else:
            tid = (a.tm_mday - 4)*24 + b.tm_hour
        reqarr[tid][vdic[items[10]]][int(items[0+lev])] += 1

    print('-- Lev = ' + str(lev+1) + ' ;' + 'Sparsity = ' + str(np.sum(reqarr == 0)/float((336-24)*1000*pow(4,lev+1))))

def creatdataset(lev):
    f = open('new_top_2000', 'r')
    fv = f.readlines()
    f.close()

    vdic = {}
    for i in range(1000):
        items = fv[i].split()
        vdic[items[1]] = i

    f = open('non_hier_sess_1000', 'r')
    fin = f.readlines()
    f.close()

    reqarr = np.zeros((336-24, 1000, pow(4,lev+1)))
    # reqarr = [[[0 for k in range(4)] for i in range(1000)] for j in range(336-24)]
    for i in range(len(fin)):
        items = fin[i].split()

        a = time.strptime(items[6], "%Y-%m-%d")
        b = time.strptime(items[7], "%H:%M:%S")
        if a.tm_mday == 10:
            continue
        elif a.tm_mday > 10:
            tid = (a.tm_mday - 4 - 1)*24 + b.tm_hour
        else:
            tid = (a.tm_mday - 4)*24 + b.tm_hour
        reqarr[tid][vdic[items[10]]][int(items[0+lev])] += 1

    
    reqarr_2 = reqarr/float(len(fin))

    winsize = 8
    x_all = np.zeros((336-24-winsize, winsize-1, 1000*pow(4,lev+1)))
    y_all = np.zeros((336-24-winsize, 1000*pow(4,lev+1)))

    for i in range(336-24-winsize):
        for j in range(i,i+7):
            x_all[i][j-i] = reqarr[j].reshape(1, 1000*pow(4,lev+1))[0]
        y_all[i] = reqarr[i+7].reshape(1, 1000*pow(4,lev+1))[0]

    f = open('lev_' + str(lev+1) + '_absolute_dataset_new_LSTM.pkl', 'w')
    pickle.dump(x_all, f)
    pickle.dump(y_all, f)
    f.close()
    print('--- absolute file has been DONE! ---')

    winsize = 8
    x_all = np.zeros((336-24-winsize, winsize-1, 1000*pow(4,lev+1)))
    y_all = np.zeros((336-24-winsize, 1000*pow(4,lev+1)))

    for i in range(336-24-winsize):
        for j in range(i,i+7):
            x_all[i][j-i] = reqarr_2[j].reshape(1, 1000*pow(4,lev+1))[0]
        y_all[i] = reqarr_2[i+7].reshape(1, 1000*pow(4,lev+1))[0]

    f = open('lev_' + str(lev+1) + '_ratio_dataset_new_LSTM.pkl', 'w')
    pickle.dump(x_all, f)
    pickle.dump(y_all, f)
    f.close()
    print('--- ratio file has been DONE!---')

    # print '--- Dataset has CREATED! ---'
    #
    # # expected input data shape: (batch_size, timesteps, data_dim)
    # timesteps = winsize - 1
    # data_dim = 4000
    #
    # model = Sequential()
    # model.add(LSTM(4000, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    # model.add(Dense(4000, activation='softmax'))
    # # model.add(LSTM(32, return_sequences=True,
    # #                input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    # # model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    # # model.add(LSTM(32))  # return a single vector of dimension 32
    # # model.add(Dense(10, activation='softmax'))
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
    #
    # x, y = shuffle(x_all, y_all, random_state=1)
    # num_training = int(0.8 * len(x))
    # num_validation = int(0.9 * len(x))
    #
    # # Generate dummy training data
    # x_train, y_train = x[:num_training], y[:num_training]
    #
    # # Generate dummy validation data
    # x_val, y_val = x[num_training:num_validation], y[num_training:num_validation]
    #
    # # Generate dummy testing data
    # x_test, y_test = x[num_validation:], y[num_validation:]
    #
    # model.fit(x_train, y_train,
    #       batch_size=64, epochs=8,
    #       validation_data=(x_val, y_val))
    # score = model.evaluate(x_test, y_test, batch_size=16)
    # print score


def train(para):
    qt = 4313869
    start_time = time.time()
    if para == 1:
        f = open('ratio_dataset_new_LSTM.pkl', 'r')
        x_all = pickle.load(f)
        y_all = pickle.load(f)
        f.close()
    elif para == 2:
        f = open('absolute_dataset_new_LSTM.pkl', 'r')
        x_all = pickle.load(f)
        y_all = pickle.load(f)
        f.close()
    end_time = time.time()
    print("Loading data need %g seconds" % (end_time - start_time))

    # normalization for x_all and y_all
    x_all = (x_all - min(x_all))/float(max(x_all) - min(x_all))
    y_all = (y_all - min(y_all))/float(max(y_all) - min(y_all))


    # expected input data shape: (batch_size, timesteps, data_dim)
    start_time = time.time()
    winsize = 8
    timesteps = winsize - 1
    data_dim = 4000

    print(len(x_all))
    print(len(x_all[0]))
    print(len(x_all[0][0]))

    model = Sequential()
    model.add(LSTM(4000, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(Dense(4000, activation='softmax'))
    # model.add(Dense(4000, input_shape=(timesteps, data_dim), activation='relu'))
    # model.add(LSTM(8000, return_sequences=True,
    #            input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    # # model.add(LSTM(8000, return_sequences=True))  # returns a sequence of vectors of dimension 32
    # model.add(LSTM(8000))  # return a single vector of dimension 32
    # model.add(Dense(4000, activation='softmax'))

    # model.add(LSTM(32, return_sequences=True,
    #                input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    # model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    # model.add(LSTM(32))  # return a single vector of dimension 32
    # model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # model.compile(loss='mean_squared_error',
    #               optimizer='sgd',
    #               metrics=['accuracy'])

    x, y = shuffle(x_all, y_all, random_state=1)
    num_training = int(0.8 * len(x))
    num_validation = int(0.9 * len(x))

    # Generate dummy training data
    x_train, y_train = x[:num_training], y[:num_training]
    print('------ x train -------')
    print(len(x_train))
    print(x_train[0])

    # Generate dummy validation data
    x_val, y_val = x[num_training:num_validation], y[num_training:num_validation]

    # Generate dummy testing data
    x_test, y_test = x[num_validation:], y[num_validation:]
    print(len(x_test))
    print(x_test[0])

    model.fit(x_train, y_train,
          batch_size=16, epochs=4,
          validation_data=(x_val, y_val))
    y_pred = model.predict(x_test, batch_size=16)
    print('------ Y -------')
    # print len(y_pred)
    # print len(y_pred[0])
    # print len(y_test)
    # print len(y_test[0])
    print(y_pred)
    print('---y_test ---')
    print(y_test)
    # score = model.evaluate(x_test, y_test, batch_size=16)
    # print score
    end_time = time.time()
    print("Processing need %g seconds" % (end_time - start_time))


def twolayerLSTM(para, lev):
    ## Part1: Load data from .pkl file
    #start_time = time.time()
    #if para == 1:
    #    f = file('lev_' + str(lev+1) + '_ratio_dataset_new_LSTM.pkl', 'r')
    #    x_all = pickle.load(f)
    #    y_all = pickle.load(f)
    #    f.close()
    #elif para == 2:
    #    f = file('lev_' + str(lev+1) + '_absolute_dataset_new_LSTM.pkl', 'r')
    #    x_all = pickle.load(f)
    #    y_all = pickle.load(f)
    #    f.close()
    #elif para == 3:
    #    f = file('lev_' + str(lev+1) + '_absolute_dataset_new_LSTM.pkl', 'r')
    #    x_all = pickle.load(f)
    #    y_all = pickle.load(f)
    #    f.close()
    #    for i in range(len(x_all)):
    #        for j in range(len(x_all[0])):
    #            for k in range(len(x_all[0][0])):
    #                if x_all[i][j][k] == 0:
    #                    x_all[i][j][k] == 0
    #                elif x_all[i][j][k] > 0:
    #                    x_all[i][j][k] = log10(x_all[i][j][k])+1
    #    for i in range(len(y_all)):
    #        for j in range(len(y_all[0])):
    #                if y_all[i][j] == 0:
    #                    y_all[i][j] == 0
    #                elif y_all[i][j] > 0:
    #                    y_all[i][j] = log10(y_all[i][j])+1
    #end_time = time.time()
    #print("Loading data need %g seconds" % (end_time - start_time))

    ## normalization for x_all and y_all
    #max_x = x_all.max(axis = 1).max(axis=1).max()
    #min_x = x_all.min(axis = 1).min(axis=1).min()
    #max_y = y_all.max(axis = 1).max()
    #min_y = y_all.min(axis = 1).min()
    #x_all = (x_all - min_x)/float(max_x - min_x)
    #y_all = (y_all - min_y)/float(max_y - min_y)

    #print len(x_all)
    #print len(x_all[0])
    #print len(x_all[0][0])
    #print len(y_all)
    #print len(y_all[0])

    ## Part 2: Creat data for train:validation:test = 8:1:1
    #x, y = shuffle(x_all, y_all, random_state=1)
    #num_training = int(0.8 * len(x))
    #num_validation = int(0.9 * len(x))
    #print '--- Dataset has been SHUFFLED! ----'

    ## Generate dummy training data
    #x_train, y_train = x[:num_training], y[:num_training]
    #f = open('experiment_datasets/''lev_' + str(lev+1) + '_training_data.pkl''', 'w')
    #pickle.dump(x_train, f)
    #pickle.dump(y_train, f)
    #f.close()
    #print '------ x train -------'
    #print len(x_train)
    ##sys.exit(2)

    ## Generate dummy validation data
    #x_val, y_val = x[num_training:num_validation], y[num_training:num_validation]
    #f = open('experiment_datasets/''lev_' + str(lev+1) + '_validation_data.pkl''', 'w')
    #pickle.dump(x_val, f)
    #pickle.dump(y_val, f)
    #f.close()
    #print '------ x val -------'
    #print len(x_val)
    ##sys.exit(2)

    ## Generate dummy testing data
    #x_test, y_test = x[num_validation:], y[num_validation:]
    #f = open('experiment_datasets/''lev_' + str(lev+1) + '_testdata.pkl''', 'w')
    #pickle.dump(x_test, f)
    #pickle.dump(y_test, f)
    #f.close()
    #sys.exit(2)
    #print '------ x test -------'
    #print len(x_test)

    # Load data from saved _datasets.pkl from ../experiment_datasets
    start_time = time.time()
    f = open('experiment_datasets/''lev_' + str(lev+1) + '_training_data.pkl''', 'r')
    x_train = pickle.load(f)
    y_train = pickle.load(f)
    f.close()

    f = open('experiment_datasets/''lev_' + str(lev+1) + '_validation_data.pkl''', 'r')
    x_val = pickle.load(f)
    y_val = pickle.load(f)
    f.close()

    f = open('experiment_datasets/''lev_' + str(lev+1) + '_testdata.pkl''', 'r')
    x_test = pickle.load(f)
    y_test = pickle.load(f)
    f.close()
    end_time = time.time()
    print('--- Loading data need ' + str(end_time - start_time) + ' seconds! ---')

    # expected input data shape: (batch_size, timesteps, data_dim)
    start_time = time.time()
    winsize = 8
    data_dim = 4000 * pow(4,lev)
    timesteps = winsize - 1

    # Part 3: initialize the model
    model = Sequential()
    model.add(TimeDistributed(Dense(data_dim/pow(4, lev)), input_shape=(timesteps, data_dim)))
    # model.add(Dropout(0.5))
    data_dim = data_dim / pow(4, lev)
    model.add(LSTM(data_dim,input_shape=(timesteps, data_dim)))
    model.add(Dense(data_dim * pow(4, lev), activation = 'relu'))

    print(model.summary())

    sgd = optimizers.SGD(lr=8, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['accuracy'])
    print('--- Model has been CREATED! ---')



    # Part 4: Fit the model
    earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
    model.fit(x_train, y_train,
          batch_size=64, epochs=500,
          validation_data=(x_val, y_val),callbacks=[earlyStopping])


    # Prediction
    y_pred = model.predict(x_test)


    # Compute the RMSE of prediction
    rms = sqrt(mean_squared_error(y_pred, y_test))
    print('Method: Seq_pred; RMSE = ' + str(rms))

    # Save the model
    model.save('lev_' + str(lev+1) + '_model.h5')
    f = open('results/''lev_' + str(lev+1) + '_results.pkl''', 'w')
    pickle.dump(y_test, f)
    pickle.dump(y_pred, f)
    f.close()


    # print the prediction result on testing data
    print('--- y_test ---')
    print(y_test)
    print('--- y_pred ---')
    print(y_pred)
    end_time = time.time()
    print("Model need %g seconds" % (end_time - start_time))



# preprocess()
# new1000()
# hiersessfile()
# creatdataset(0)
# train(1)
#twolayerLSTM(3,0)

# for i in range(2,3):
#     twolayerLSTM(3,i)
#     print '---iteration ' + str(i) + ' HAVE DONE!---'

# hiersessfile()
# for lev in range(1,5):
#     creatdataset(lev)
# sparsity(0)
# creatdataset(4)

## 2021.10.27

ROOT_DIR = '/tmp/pycharm_project_618/'
results = []
data_path = ROOT_DIR + 'experiment_datasets/' + os.listdir(ROOT_DIR + 'experiment_datasets/')[0]

lev = 0
print('Task: Local Popularity Prediction on Partition Level ' + str(lev+1) + '...')
print('Loading data...{}'.format(data_path))
start_time = time.time()
f = open(data_path, 'rb')
x_test = pickle.load(f, encoding='bytes')
y_test = pickle.load(f, encoding='bytes')
f.close()
end_time = time.time()
data_load_time = str(end_time - start_time)
print('Data loaded using ' + str(end_time - start_time) + ' seconds.')
print('Start prediction...')
start_time = time.time()
loaded_model = tf.keras.models.load_model(ROOT_DIR + 'lev_' + str(lev+1) + '_model.h5')
y_pred = loaded_model.predict(x_test, batch_size=16)
end_time = time.time()
score = loaded_model.evaluate(x_test, y_test, batch_size=16)
# print('---y_pred ---')
# print(y_pred)
# print('---y_test ---')
# print(y_test)
prediction_time = str(end_time - start_time)
print('Prediciton done using ' + str(end_time - start_time) + ' seconds.')
print('Statistics: ')
mse = str(mean_squared_error(y_test,y_pred))
rmse = str(sqrt(mean_squared_error(y_test, y_pred)))
mae = str(mean_absolute_error(y_test,y_pred))
print(">> MSE: " + mse)
print(">> RMSE: " + rmse)
print(">> MAE: " + mae)

results.append({
    "type": "image",
    "title": "Cumulative Distribution of MAE for Contents",
    "description": "Cumulative Distribution of MAE for Contents",
    "data": "avgregion_lev_1.png"
})
results.append({
    "type": "table",
    "title": "Statistics Report",
    "data": {
        "thead": ['Metrics', 'Values'],
        "tbody": [
            ['Data loaded time', data_load_time],
            ['Prediction time', prediction_time],
            ['Mean Squared Error', mse],
            ['Root Mean Squared Error', rmse],
            ['Mean Absolute Error', mae]
        ]
    }
})

# print("Evaluate Score:" + str(score))

print(results)
with open(ROOT_DIR + 'result/result.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False)

# with open(result_path + 'result.json', 'r', encoding='utf-8') as f1:
#     res = json.load(f1)

# print(res)

