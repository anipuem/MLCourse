from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from loadData import dataLoader, dataLoaderwithHeader
from loadData import transftoArray
import seaborn as sns
import matplotlib.pyplot as plt
import fitter
from loadData import WriteCSV
from loadData import flatDict
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def loaddata():
    # data pre-load
    trainXDir = './data1forEx1to4/train1_icu_data.csv'
    trainYDir = './data1forEx1to4/train1_icu_label.csv'
    trainX, trainY, headerX, headerY = dataLoaderwithHeader(trainXDir, trainYDir, mode='train')
    AtrainX, AtrainY = transftoArray(trainX, trainY)
    testXDir = './data1forEx1to4/test1_icu_data.csv'
    testYDir = './data1forEx1to4/test1_icu_label.csv'
    testX, testY = dataLoader(testXDir, testYDir, mode='test')
    AtestX, AtestY = transftoArray(testX, testY)
    return AtrainX, AtrainY, AtestX, AtestY, headerX, headerY


def split_to_trainVal(AtrainX, AtrainY):
    scaler = StandardScaler()  # standardization
    x_std = scaler.fit_transform(AtrainX)

    x_train, x_val, y_train, y_val = train_test_split(x_std, AtrainY, test_size=.2)
    print('train x has #', len(x_train), '; validation x has #', len(x_val),
          '; train y has #', len(y_train), '; validation y has #', len(y_val))
    return x_train, x_val, y_train, y_val


def histfig(headerX, x_train):
    counter = 0
    plt.figure(figsize=(16, 7))
    for i in range(len(headerX)):  # len(headerX)
        counter += 1
        print(counter)
        if counter < 10:
            plt.subplot(2, 5, (i+1) % 10)
            sns.distplot(x_train[:, i])
            plt.xlabel(headerX[i])
            if i == len(headerX)-1:
                plt.margins(0, 0)
                plt.subplots_adjust(top=0.96, bottom=0.1, right=0.98, left=0.04, hspace=0.3, wspace=0.3)
                if os.path.exists('.\disfig'):
                    pass
                else:
                    os.mkdir('.\disfig')
                plt.savefig('.\disfig\plot%d.jpg' % i, dpi=100)
        else:
            plt.subplot(2, 5, 10)
            sns.distplot(x_train[:, i])
            plt.xlabel(headerX[i])
            plt.margins(0, 0)
            plt.subplots_adjust(top=0.96, bottom=0.1, right=0.98, left=0.04, hspace=0.3, wspace=0.3)
            if os.path.exists('.\disfig'):
                pass
            else:
                os.mkdir('.\disfig')
            plt.savefig('.\disfig\plot%d.jpg' % i, dpi=100)
            counter = 0
            plt.figure(figsize=(16, 7))


def distStore(x_train, headerX):
    bestDist = []
    typeList = []
    context = []
    counter = []
    for i in range(len(x_train[0])):  # len(x_train)
        f = fitter.Fitter(x_train[:, i], distributions=['lognorm', 'bernoulli', 'burr', 'norm', 'poisson',
                                                        'gamma', 'gennorm', 'cauchy', 'gaussian', 'uniform',
                                                        'genhyperbolic', 'laplace', 'chi2', 'expon'], timeout=100)
        f.fit()
        bestDist.append(f.get_best(method='sumsquare_error'))
        keydistr, _ = flatDict(f.get_best(method='sumsquare_error'))
        if keydistr in typeList:
            num = typeList.index(keydistr)
            context[num].append(headerX[i])
            counter[num] += 1
        else:
            typeList.append(keydistr)
            context.append([headerX[i]])
            counter.append(1)
        print(str(i + 1) + ':')
        print(f.get_best(method='sumsquare_error'))
        # For visible
        # f.hist()
        # f.plot_pdf(Nbest=5, lw=2, method='sumsquare_error')
        # plt.show()
    WriteCSV('Distnote', ['Name', 'ParamList'], bestDist, type='dict')
    print('Totally ' + str(len(counter)) + ' types distribution')
    for i in range(len(counter)):
        print(i+1, ': ', typeList[i], context[i], counter[i])


if __name__ == '__main__':
    # load data
    AtrainX, AtrainY, AtestX, AtestY, headerX, headerY = loaddata()
    # split to training and validation
    x_train, x_val, y_train, y_val = split_to_trainVal(AtrainX, AtrainY)
    # histfig(headerX, x_train)
    # distStore(x_train, headerX)
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    print('-'*10, 'Training', '-'*10)
    print("Error rate of training: %lf" % (1-clf.score(x_train, y_train)))
    print("Error rate of validation: %lf" % (1-clf.score(x_val, y_val)))
    print('-' * 10, 'Testing', '-' * 10)
    scaler = StandardScaler()  # standardization
    x_testnorm = scaler.fit_transform(AtestX)
    print("Error rate of Testing: %lf" % (1 - clf.score(x_testnorm, AtestY)))
    print(clf.predict_proba(x_testnorm))
    counterTest = 0
    for i in range(len(AtestY)):
        if clf.predict_proba(x_testnorm)[i][0]>10*clf.predict_proba(x_testnorm)[i][1]:
            y_predict = 0
        else:
            y_predict = 1
        if y_predict != AtestY[i]:
            counterTest += 1
    print('Error rate of minimum-risk Bayesian decision: %lf' % (counterTest/len(AtestY)))
