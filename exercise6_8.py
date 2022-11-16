from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from loadData import dataLoader
from loadData import transftoArray
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def loaddata():
    # data pre-load
    trainXDir = './data1forEx1to4/train1_icu_data.csv'
    trainYDir = './data1forEx1to4/train1_icu_label.csv'
    trainX, trainY = dataLoader(trainXDir, trainYDir, mode='train')
    AtrainX, AtrainY = transftoArray(trainX, trainY)
    testXDir = './data1forEx1to4/test1_icu_data.csv'
    testYDir = './data1forEx1to4/test1_icu_label.csv'
    testX, testY = dataLoader(testXDir, testYDir, mode='test')
    AtestX, AtestY = transftoArray(testX, testY)
    return AtrainX, AtrainY, AtestX, AtestY


def split_to_trainVal(AtrainX, AtrainY):
    scaler = StandardScaler()  # standardization
    x_std = scaler.fit_transform(AtrainX)

    x_train, x_val, y_train, y_val = train_test_split(x_std, AtrainY, test_size=.2)
    print('train x has #', len(x_train), '; validation x has #', len(x_val),
          '; train y has #', len(y_train), '; validation y has #', len(y_val))
    return x_train, x_val, y_train, y_val


def lasso(x_train, x_val, y_train, y_val, mode='LassoLarsCV'):
    if mode=='LassoLarsCV':
        model = LassoLarsCV()  # get alpha automatically
        model.fit(x_train, y_train)
        print('Best alpha：', model.alpha_)
    elif mode=='Lasso':
        model = Lasso(alpha=0.01)
        model.fit(x_train, y_train)
    else:
        model = LassoCV()  # get alpha automatically
        model.fit(x_train, y_train)
        print('Best alpha：', model.alpha_)

    counter = np.sum(model.coef_ != 0)
    print('Non-zero feature after LASSO has the number of ', 108-counter)
    print('-'*10, 'Start training', '-'*10)
    print('-' * 10, 'When using LASSO', '-' * 10)
    print('Training error is ', 1.0-model.score(x_train, y_train))
    print('Validation error is', 1.0-model.score(x_val, y_val))
    return model


def test(x_test, y_test, predictor):
    scaler = StandardScaler()  # standardization
    x_stdtest = scaler.fit_transform(x_test)
    acc = predictor.score(x_stdtest, y_test)
    print('Testing error is', 1.0 - acc)


if __name__ == '__main__':
    # load data
    AtrainX, AtrainY, AtestX, AtestY = loaddata()
    # split to training and validation
    x_train, x_val, y_train, y_val = split_to_trainVal(AtrainX, AtrainY)
    # feature extraction
    # train a model and test
    predictor = lasso(x_train, x_val, y_train, y_val, 'LassoLarsCV')  # three types
    test(AtestX, AtestY, predictor)
