import pickle

for i in range(10):

    with open('/tmp/pycharm_project_704/data/pro_test_flows_' + str(i) + '_noip_fold.pkl', 'rb') as f:
        tmp = pickle.load(f)

    for key in tmp['test'].keys():
        tmp['test'][key] = tmp['test'][key][:500]

    with open('/tmp/pycharm_project_704/data_new/pro_test_flows_' + str(i) + '_noip_fold.pkl', 'wb') as f1:
        pickle.dump(tmp, f1)

