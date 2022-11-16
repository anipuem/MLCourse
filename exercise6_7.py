from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from loadData import dataLoader
from loadData import transftoArray
from sklearn.decomposition import PCA
import numpy as np


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


def pca(x, n_components):
    # PCA
    print('-'*10, 'Start PCA', '-'*10)
    print('The feature number used to be ', x.shape[1])
    pcamodel = PCA(n_components=0.9)  # 保证降维后的数据保持90%的信息
    pcamodel.fit(x)
    xtrain = pcamodel.fit_transform(x)
    print('The feature number after PCA is ', xtrain.shape[1])
    return xtrain, pcamodel


def svm(x_train, x_val, y_train, y_val, pcamodel):
    print('-'*10, 'Start training', '-'*10)
    predictor = SVC(gamma='scale', C=1, decision_function_shape='ovo', kernel='rbf', class_weight='balanced', probability=True)
    predictor.fit(x_train, y_train)
    x_valpca = pcamodel.transform(x_val)
    print('-' * 10, 'When using Gaussian kernel', '-' * 10)
    print('Training error is ', 1.0-predictor.score(x_train, y_train))
    print('Validation error is', 1.0-predictor.score(x_valpca, y_val))
    return predictor


def test(x_test, y_test, predictor, pcamodel):
    scaler = StandardScaler()  # standardization
    x_stdtest = scaler.fit_transform(x_test)
    x_testpca = pcamodel.transform(x_stdtest)
    acc = predictor.score(x_testpca, y_test)
    print('Testing error is', 1.0 - acc)


if __name__ == '__main__':
    # load data
    AtrainX, AtrainY, AtestX, AtestY = loaddata()
    # split to training and validation
    x_train, x_val, y_train, y_val = split_to_trainVal(AtrainX, AtrainY)
    # feature extraction
    x_trainpca, pcamodel = pca(x_train, n_components=0.9)
    # train a model and test
    predictor = svm(x_trainpca, x_val, y_train, y_val, pcamodel)
    test(AtestX, AtestY, predictor, pcamodel)
