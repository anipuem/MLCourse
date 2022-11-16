import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def dataLoader(xPath, yPath, mode='train'):
    # load trainX
    trainX = []
    with open(xPath) as xf:
        xf_csv = csv.reader(xf)
        # get headers
        headersX = next(xf_csv)
        # get x values
        for xrow in xf_csv:
            trainX.append(xrow)
        print('-------------------------------Start loading X-------------------------------------')
        print('There are totally %d samples for %sing' % (len(trainX), mode))
    # load trainY
    trainY = []
    with open(yPath) as yf:
        yf_csv = csv.reader(yf)
        # get headers
        headersY = next(yf_csv)
        print('-------------------------------Start loading Y-------------------------------------')
        for yrow in yf_csv:
            trainY.append(yrow)
        if len(trainY) == len(trainX):
            print('Finish loading %d samples' % len(trainY))
        else:
            print('Index error: X and Y have different sample size.')
    return trainX, trainY


def dataLoaderwithHeader(xPath, yPath, mode='train'):
    # load trainX
    trainX = []
    with open(xPath) as xf:
        xf_csv = csv.reader(xf)
        # get headers
        headersX = next(xf_csv)
        # get x values
        for xrow in xf_csv:
            trainX.append(xrow)
        print('-------------------------------Start loading X-------------------------------------')
        print('There are totally %d samples for %sing' % (len(trainX), mode))
    # load trainY
    trainY = []
    with open(yPath) as yf:
        yf_csv = csv.reader(yf)
        # get headers
        headersY = next(yf_csv)
        print('-------------------------------Start loading Y-------------------------------------')
        for yrow in yf_csv:
            trainY.append(yrow)
        if len(trainY) == len(trainX):
            print('Finish loading %d samples' % len(trainY))
        else:
            print('Index error: X and Y have different sample size.')
    return trainX, trainY, headersX, headersY


def transftoArray(trainX, trainY):
    AtrainX = np.array(trainX, dtype='float64')
    AtrainY = np.array(trainY, dtype='float64')
    AtrainY = AtrainY.reshape(len(trainY), )
    # make sure the number of 0,1 is not a string but a integer
    for i in range(len(trainY)):
        AtrainY[i] = int(AtrainY[i])
    return AtrainX, AtrainY


def findK(AtrainX, AtrainY):
    # learn best K
    scores = []
    ks = []
    x_train, x_validation, y_train, y_validation = train_test_split(AtrainX, AtrainY, test_size=0.2, random_state=2020)
    print('-------------------------------Start finding the best k-------------------------------')
    for i in range(1, 55):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)  # 训练模型
        score = knn.score(x_validation, y_validation)  # 计算模型预测准确率
        ks.append(i)
        scores.append(score)
    scores_arr = np.array(scores)
    ks_arr = np.array(ks)
    sortScore = np.argsort(-scores_arr)
    plt.plot(ks_arr, scores_arr)
    plt.xlabel('k_value')
    plt.ylabel('score')
    plt.title('The score plot among k=[1 to 55]')
    plt.show()
    return ks_arr[sortScore[0]] - 1


def WriteCSV(csvPath, headerName, List, type='list'):
    path = csvPath  # 'Distnote'
    with open('{}.csv'.format(path), 'w', encoding='utf-8', newline='') as f:
        csv_write = csv.writer(f)
        # Header
        csv_head = headerName  # ["good", "bad"]
        csv_write.writerow(csv_head)
        if type == 'list':
            for tmp in List:
                csv_write.writerow(tmp)
        elif type == 'dict':
            for item in List:
                key, value = flatDict(item)
                csv_write.writerow([key, value])
        else:
            print('Type not define')


def flatDict(Dict):
    if len(Dict.keys()) == 1:
        for thing in Dict.keys():
            key = thing
            return key, Dict[key]
    else:
        raise Exception("flatDict仅能处理单Key字典")
