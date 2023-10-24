import csv
import json

PARAS_BQR = {
    'PARAS': {'Port': '11106', '负载数量': '100', '负载大小': '1472', '负载速率': '1500', '重试次数': '5'},
    'MODEL_PATH': '/tmp/pycharm_project_237/abw-project/image/wangluotuopu.png',
    'DATA_PATH': '/tmp/pycharm_project_237/abw-project/datas'
}


if __name__ == '__main__':
    results = []
    RE_BQR, COST_BQR, TIME_BQR = [], [], []
    ROOT_DIR = '/tmp/pycharm_project_237/abw-project/'
    METHOD_BQR = ['BQR', 'ASSOLO', 'PTR', 'pathload', 'Spruce']
    print('RUN BQR...')

    with open(PARAS_BQR['DATA_PATH'] + '/相对误差.csv') as f:
        reader = csv.reader(f)
        for i in range(11):
            RE_BQR += [next(reader)]
            RE_BQR[i] = [(float(x) if i > 0 else x) for x in RE_BQR[i]]
        print(RE_BQR)
    with open(PARAS_BQR['DATA_PATH'] + '/包开销.csv') as f:
        reader = csv.reader(f)
        for i in range(11):
            COST_BQR += [next(reader)]
            COST_BQR[i] = [(float(x) if i > 0 else x) for x in COST_BQR[i]]
        print(COST_BQR)
    with open(PARAS_BQR['DATA_PATH'] + '/测量时间.csv') as f:
        reader = csv.reader(f)
        for i in range(11):
            TIME_BQR += [next(reader)]
            TIME_BQR[i] = [(float(x) if i > 0 else x) for x in TIME_BQR[i]]
        print(TIME_BQR)

    results.append({
        "type": "image",
        "title": "Network topology",
        "description": "Null",
        "data": "wangluotuopu.png"
    })

    results.append({
        "type": "table",
        "title": "Packet cost",
        "data": {
            "thead": COST_BQR[0],
            "tbody": COST_BQR[1:]
        }
    })

    results.append({
        "type": "image",
        "title": "Packet cost",
        "description": "Null",
        "data": "baokaixiao.png"
    })

    results.append({
        "type": "table",
        "title": "Measurement time",
        "data": {
            "thead": TIME_BQR[0],
            "tbody": TIME_BQR[1:]
        }
    })

    results.append({
        "type": "image",
        "title": "Measurement time",
        "description": "Null",
        "data": "celiangshijian.png"
    })

    results.append({
        "type": "table",
        "title": "Relative error",
        "data": {
            "thead": RE_BQR[0],
            "tbody": RE_BQR[1:]
        }
    })

    results.append({
        "type": "image",
        "title": "Relative error",
        "description": "Null",
        "data": "xiangduiwucha.png"
    })

    print(results)
    with open(ROOT_DIR + 'result/result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

    # with open(result_path + 'result.json', 'r', encoding='utf-8') as f1:
    #     res = json.load(f1)

    # print(res)