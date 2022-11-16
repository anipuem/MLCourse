# As the requirment says "Find public packages for Perceptron and Logistic Regression (LR)"
# import more libraries
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import csv
import numpy as np

# load trainX
train1X = []
print('For feature x:')
with open('./data1forEx1to4/train1_icu_data.csv') as x1f:
    x1f_csv = csv.reader(x1f)
    # get headers
    headers1X = next(x1f_csv)
    print('-------------------------------feature counting-------------------------------------')
    print('Totally %d features' % len(headers1X),'included:\n', headers1X)
    # get x values
    for x1row in x1f_csv:
        train1X.append(x1row)
    print('-------------------------------sample counting-------------------------------------')
    print('There are totally %d samples for training' % len(train1X))
# load trainY
train1Y = []
print('For label y:')
with open('./data1forEx1to4/train1_icu_label.csv') as y1f:
    y1f_csv = csv.reader(y1f)
    # get headers
    headers1Y= next(y1f_csv)
    print('-------------------------------label counting-------------------------------------')
    print('Totally %d label' % len(headers1Y),'included:\n', headers1Y)
    for y1row in y1f_csv:
        train1Y.append(y1row)
    print('-------------------------------sample counting-------------------------------------')
    if len(train1Y)==len(train1X):
        print('There are totally %d samples for training' % len(train1Y), '，which has the same counting result as X')
    else:
        print('Index error: X and Y have different sample size.')
# load X features
test1X = []
with open('./data1forEx1to4/test1_icu_data.csv') as test1xf:
    t1xf_csv = csv.reader(test1xf)
    # get headers
    theaders1X = next(t1xf_csv)
    # get X values
    for t1xrow in t1xf_csv:
        test1X.append(t1xrow)
# load Y features
test1Y = []
with open('./data1forEx1to4/test1_icu_label.csv') as test1yf:
    t1yf_csv = csv.reader(test1yf)
    # get headers
    theaders1Y= next(t1yf_csv)
    # get Y value
    for t1yrow in t1yf_csv:
        test1Y.append(t1yrow)
    if len(test1Y)==len(test1X):
        print('There are totally %d samples for testing' % len(test1Y))
    else:
        print('Index error: X and Y have different sample size.')
print('Test data load successfully')

# define a Perceptron
clf1 = Perceptron(fit_intercept=False, max_iter=2000, shuffle=True)
# train a Perceptron
Atrain1X = np.array(train1X, dtype = 'float64')
Atrain1Y = np.array(train1Y, dtype = 'float64')
clf1.fit(train1X, train1Y)
# training result of w* and b
W = clf1.coef_[0]
b = clf1.intercept_

Atest1X = np.array(test1X, dtype = 'float64')
Atest1Y = np.array(test1Y, dtype = 'float64')
for i in range(len(test1Y)):
    Atest1Y[i][0]=int(Atest1Y[i][0])
# ypredict = clf1.predict(Atest1X)
# print(accuracy_score(ypredict, Atest1Y))
print(clf1.score(Atrain1X, Atrain1Y, W))

# cross validation
Atrain1X = np.array(train1X, dtype = 'float64')
Atrain1Y = np.array(train1Y, dtype = 'float64')
# make sure the number of 0,1 is not a string but a integer
for i in range(len(train1Y)):
    Atrain1Y[i]=int(Atrain1Y[i])
scores = cross_val_score(clf1, Atrain1X, Atrain1Y, cv=5, scoring='accuracy')
errorMean = 1-np.mean(scores)
print('There are totally 5-fold cross validation, cross validation scores are ', scores, 'independently')
print('Therefore, the mean cross validation error is %.5f.' % errorMean)
