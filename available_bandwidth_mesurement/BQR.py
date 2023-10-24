import csv

PARAS_BQR = {
    'PARAS': {'Port': '11106', '负载数量': '100', '负载大小': '1472', '负载速率': '1500', '重试次数': '5'},
    'MODEL_PATH': '/tmp/pycharm_project_159/abw-project/image/wangluotuopu.png',
    'DATA_PATH': '/tmp/pycharm_project_159/abw-project/data',
    'SPEED': '100'
}


def BQR():
    RE_BQR, COST_BQR, TIME_BQR = [], [], []
    OUTPUT_BQR = ''
    METHOD_BQR = ['BQR', 'ASSOLO', 'PTR', 'pathload', 'Spruce']
    print('RUN BQR...')

    with open(PARAS_BQR['DATA_PATH'] + '/相对误差.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for i in range(10):
            RE_BQR += [next(reader)]
            RE_BQR[i] = [float(x) for x in RE_BQR[i]]
        print(RE_BQR)
    with open(PARAS_BQR['DATA_PATH'] + '/包开销.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for i in range(10):
            COST_BQR += [next(reader)]
            COST_BQR[i] = [float(x) for x in COST_BQR[i]]
        print(COST_BQR)
    with open(PARAS_BQR['DATA_PATH'] + '/测量时间.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for i in range(10):
            TIME_BQR += [next(reader)]
            TIME_BQR[i] = [float(x) for x in TIME_BQR[i]]
        print(TIME_BQR)

    speed = int(int(PARAS_BQR['SPEED']) / 100)
    bias = int(PARAS_BQR['SPEED']) - speed * 100
    _re = [0, 0, 0, 0, 0, 0]
    _time = [0, 0, 0, 0, 0, 0]
    _cost = [0, 0, 0, 0, 0, 0]
    OUTPUT_BQR += ('Speed: ' + PARAS_BQR['SPEED'] + 'Mbps: \n')
    OUTPUT_BQR += 'Method  Relative Error(%)  Time(ms)  Packet Cost(MB)\n'
    for i in range(6):
        _re[i] = RE_BQR[speed][i] + (RE_BQR[speed + 1][i] - RE_BQR[speed][i]) * bias / 100
        _time[i] = TIME_BQR[speed][i] + (TIME_BQR[speed + 1][i] - TIME_BQR[speed][i]) * bias / 100
        _cost[i] = COST_BQR[speed][i] + (COST_BQR[speed + 1][i] - COST_BQR[speed][i]) * bias / 100
        if i > 0:
            OUTPUT_BQR += (
                    METHOD_BQR[i - 1] + '\t%.2f\t\t%.2f  \t%.2f\n' % (_re[i], _time[i], _cost[i]))
    print(_re)
    print(_time)
    print(_cost)


if __name__ == '__main__':
    BQR()