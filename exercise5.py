from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from loadData import dataLoader
from loadData import transftoArray


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


def svm(x_train, x_val, y_train, y_val):
    print('-'*10, 'Start training', '-'*10)
    predictor = SVC(gamma='scale', C=1.2, decision_function_shape='ovr', kernel='linear')
    predictor.fit(x_train, y_train)

    # xval_predic = predictor.predict(x_val)
    # xtrain_predic = predictor.predict(x_train)
    print('-' * 10, 'When using linear kernel', '-' * 10)
    print('Training error is ', 1.0-predictor.score(x_train, y_train))
    print('Validation error is', 1.0-predictor.score(x_val, y_val))

    # print(predictor.support_vectors_)
    # print(predictor.n_support_)  # number of support vector
    return predictor


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
    # train a model and test
    predictor = svm(x_train, x_val, y_train, y_val)
    test(AtestX, AtestY, predictor)

