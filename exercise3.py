from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from loadData import dataLoader
from loadData import transftoArray
from loadData import findK

# data pre-load
trainXDir = './data1forEx1to4/train1_icu_data.csv'
trainYDir = './data1forEx1to4/train1_icu_label.csv'
trainX, trainY = dataLoader(trainXDir, trainYDir)
AtrainX, AtrainY = transftoArray(trainX, trainY)
k = findK(AtrainX, AtrainY)
print('The best k is ', k)
testXDir = './data1forEx1to4/test1_icu_data.csv'
testYDir = './data1forEx1to4/test1_icu_label.csv'
testX, testY = dataLoader(testXDir, testYDir)
AtestX, AtestY = transftoArray(trainX, trainY)


def train(k):
    # define a classifier
    clf = KNeighborsClassifier(n_neighbors=k)
    # train a classifier
    clf.fit(AtrainX, AtrainY)

    # testing
    test_predictions = clf.predict(AtestX)
    print('Testing result is shown below: ')
    print('Accuracy:', accuracy_score(AtestY, test_predictions))
    print('MSE:', mean_squared_error(AtestY, test_predictions))


train(k)

