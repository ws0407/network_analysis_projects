import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def loadDataSet_path(datasets_path):  # 0 means all
    import os
    datasets = os.listdir(datasets_path)
    return datasets


def divide_into_group(data_all):
    import re

    diff_group = []
    len_group = []
    ann_longer_group = []
    ann_shorter_group = []
    ls = data_all.columns
    for col in ls:
        if re.search("diff_\d+", col) != None:
            diff_group.append(re.search("diff_\d+", col).string)
        elif re.search("len_path\d+", col) != None:
            len_group.append(re.search("len_path\d+", col).string)
        elif re.search("ann_longer_\d+", col) != None:
            ann_longer_group.append(re.search("ann_longer_\d+", col).string)
        elif re.search("ann_shorter_\d+", col) != None:
            ann_shorter_group.append(re.search("ann_shorter_\d+", col).string)
    return diff_group, len_group, ann_longer_group, ann_shorter_group


def measure_datasets():
    datasets_path = '../datasets/datasets/'
    # load files path
    datasets_files = loadDataSet_path(datasets_path)
    # spilt files to test and train
    data_all = pd.DataFrame()
    half_window = 15
    print(half_window)
    # train_data
    for data_file in datasets_files:
        try:
            temp = pd.read_csv(datasets_path + data_file, index_col=0)
            temp = temp.drop(
                columns=['time', 'new_sub_prefix', 'MOAS_AS', 'Victim_AS', 'MOAS', 'withdraw_unique_prefix'],
                axis=1)
            feature_sum = temp.iloc[120 - half_window + 1:120 + half_window].sum()
            feature_sum['label_0'] = temp['label_0'].iloc[120]
            data_all = data_all.append(feature_sum, ignore_index=True)
        except:
            print(datasets_path + data_file)

    datasets_path2 = '../datasets/legitimate/'
    datasets_files2 = loadDataSet_path(datasets_path2)
    for data_file in datasets_files2:
        try:
            temp = pd.read_csv(datasets_path2 + data_file, index_col=0)
            temp = temp.drop(
                columns=['time', 'new_sub_prefix', 'MOAS_AS', 'Victim_AS', 'MOAS', 'withdraw_unique_prefix'],
                axis=1)
            for i in range(0, 1440, 30):
                feature_sum = temp.iloc[i:i + 30].sum()
                data_all = data_all.append(feature_sum, ignore_index=True)
        except:
            print(datasets_path2 + data_file)
    data_all.fillna(0, inplace=True)
    return data_all


def measure_prefix(data_all):
    Prefix_Features = ['MOAS_prefix_num', 'MOAS_num', 'new_MOAS', 'new_prefix_num', 'label_0']
    p = data_all[Prefix_Features]
    p.to_csv('../result_doc/Prefix_Feature.csv')
    return p


def measure_volume(data_all):
    Prefix_Features = ['MOAS_Ann_num', 'own_Ann_num', 'duplicate_ann', 'withdraw_unique_prefix_num', 'withdraw_num',
                       'Diff_Ann', 'label_0']
    p = data_all[Prefix_Features]
    p.to_csv('../result_doc/Volume_Feature.csv')
    return p


def measure_edge():
    data_all = measure_datasets()
    print(data_all)
    diff_group, len_group, ann_longer_group, ann_shorter_group = divide_into_group(data_all)
    diff_group = sort_group(diff_group, '_', 1)
    len_group = sort_group(len_group, 'h', 1)
    ann_longer_group = sort_group(ann_longer_group, '_', 2)
    ann_shorter_group = sort_group(ann_shorter_group, '_', 2)
    data0 = data_all[data_all['label_0'] == 0]
    data1 = data_all[data_all['label_0'] == 1]
    data2 = data_all[data_all['label_0'] == 2]
    data3 = data_all[data_all['label_0'] == 3]
    data4 = data_all[data_all['label_0'] == 4]
    data5 = data_all[data_all['label_0'] == 5]
    group_name = ['Diff', 'len', 'longer', 'shorter']
    Event_Name = ['Normal', 'Prefix_Hijack', 'Route_Leak', 'Breakout', 'Fake_route', 'Defcon']
    groups = [diff_group, len_group, ann_longer_group, ann_shorter_group]
    i = 0
    flags = ['_', 'h', '_', '_']
    nums = [1, 1, 2, 2]
    for group in groups:
        num_group = []
        for pl in group:
            num_group.append(pl.split(flags[i])[nums[i]])
        a = data0[group].quantile(0.9)
        b = data1[group].quantile(0.9)
        c = data2[group].quantile(0.9)
        d = data3[group].quantile(0.9)
        e = data4[group].quantile(0.9)
        f = data5[group].quantile(0.9)
        p = pd.DataFrame(columns=Event_Name)
        p = p.reindex(a.index)
        print(a)
        p['Normal'] = a
        p['Prefix_Hijack'] = b
        p['Route_Leak'] = c
        p['Breakout'] = d
        p['Fake_route'] = e
        p['Defcon'] = f
        p.index = num_group
        print(p)
        p.to_csv('../result_doc/' + group_name[i] + '.csv')
        i += 1


def bigger(i, j, flag, num):
    return (int(i.split(flag)[num]) - int(j.split(flag)[num])) > 0


def sort_group(group, flag, num):
    for i in range(len(group)):
        for j in range(0, len(group) - i - 1):
            if bigger(group[j], group[j + 1], flag, num):
                temp = group[j]
                group[j] = group[j + 1]
                group[j + 1] = temp
    return group


def plot_box(features, flag):
    fig = plt.figure()

    data0 = features[features['label_0'] == 0]
    data1 = features[features['label_0'] == 1]
    data2 = features[features['label_0'] == 2]
    data3 = features[features['label_0'] == 3]
    data4 = features[features['label_0'] == 4]
    data5 = features[features['label_0'] == 5]

    columns = []
    title = ''
    xlabel = ''
    top = 0
    datax = data0
    save_path = ''

    if flag == 'Prefix_Hijacking':
        columns = ['MOAS_prefix_num', 'MOAS_num', 'new_MOAS', 'new_prefix_num']
        title = 'Prefix Hijacking Event'
        xlabel = 'Prefix Feature Set'
        top = 24577
        datax = data1
        save_path = '../result_doc/box1.png'
    elif flag == 'Breakout':
        columns = ['MOAS_Ann', 'Own_Ann', 'Dup_Ann', 'Wd_unique', 'Withdraw', 'Diff_Ann']
        title = 'Breakout Event'
        xlabel = 'AS Volume Feature Set'
        top = 9433232
        datax = data3
        save_path = '../result_doc/box2.png'

    ax1 = fig.add_subplot(211)
    ax1.set_xlabel(xlabel, fontsize=30)
    ax1.set_ylabel('Number', fontsize=30)
    ax1.set_title('Legitimate Event', fontsize=30)
    ax1.set_yscale('log')

    ax1.set_ylim(top=top)
    data0.drop(columns='label_0', inplace=True)
    data0.boxplot(ax=ax1)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(30)
    ax1.set_xticklabels(labels=columns)
    ax2 = fig.add_subplot(212)
    ax2.set_title(title, fontsize=30)
    ax2.set_xlabel(xlabel, fontsize=30)
    ax2.set_ylabel('Number', fontsize=30)
    ax2.set_yscale('log')
    datax.drop(columns='label_0', inplace=True)
    datax.boxplot(ax=ax2)
    ax2.set_xticklabels(labels=columns)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontsize(30)
    fig.set_size_inches(19.2, 9.61, forward=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)


def plot_line(file, xlabel):
    file_path = '../result_doc/' + file + '.csv'
    save_path = '../result_doc/' + file + '.png'

    df = pd.read_csv(file_path, index_col=0)
    df.dropna(axis=1, inplace=True)

    x = df["Normal"].index
    fig = plt.figure('small_fct')
    ax = plt.subplot(1, 1, 1)
    # goldenrod
    plt.plot(x, df["Normal"], linewidth=4, color='blue', label='Normal', linestyle='-.', ms=12)
    plt.plot(x, df["Prefix_Hijack"], linewidth=4, color='darkorange', label='Prefix_Hijack', linestyle='dashed', ms=12)
    plt.plot(x, df["Route_Leak"], linewidth=4, color='green', label='Route_Leak', ms=12)
    plt.plot(x, df["Breakout"], linewidth=4, color='orange', label='Breakout', linestyle=':', ms=12)
    plt.plot(x, df["Fake_route"], linewidth=4, color='red', label='Fake_route', linestyle='-', ms=12)
    plt.plot(x, df["Defcon"], linewidth=4, color='purple', label='Defcon', linestyle='dotted', ms=12)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_color('grey')  # 删除右边缘黑框
    ax.spines['top'].set_color('grey')  # 删除上边缘黑框
    ax.xaxis.set_tick_params(width=1.5)
    ax.yaxis.set_tick_params(width=1.5)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Number of paths(90th)', fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylim(0, 2000)
    plt.xlim(0, 60)
    plt.legend(loc='best', fontsize=14, ncol=2)
    plt.grid(linestyle=':', color='gray', zorder=1)
    fig.set_size_inches(7, 5, forward=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


data_all = measure_datasets()
data_prefix = measure_prefix(data_all)
data_volume = measure_volume(data_all)
measure_edge()
plot_box(data_prefix, 'Prefix_Hijacking')
plot_box(data_volume, 'Breakout')
file_names = ['Diff', 'len', 'longer', 'shorter']
labels = ['Path length', 'Edit distance size', 'Path increase size', 'Path decrease size']
for (file, label) in zip(file_names, labels):
    plot_line(file, label)
