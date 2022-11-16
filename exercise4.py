import numpy as np
import math
import pickle
from loadData import dataLoader
from loadData import transftoArray


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_deriv(z):
    # derivative of sigmoid
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    # derivative of ReLU
    z_ = np.copy(z)
    z_[z > 0] = 1
    z_[z < 0] = 0
    z_[z == 0] = 0.5
    return z_


def mean_squared_loss(z, y_true):
    """
        均方误差损失函数
        :param y_predict: 预测值,shape (N,d)，N为批量样本数
        :param y_true: Ground truth
        :return: loss, derivative of loss according to output
        """
    y_predict = z
    loss = np.mean(np.mean(np.square(y_predict - y_true), axis=-1))  # loss value
    dy = 2 * (y_predict - y_true) * sigmoid_deriv(z) / y_true.shape[0]  # derivative of loss
    # y_predict = relu(z)
    # loss = np.mean(np.mean(np.square(y_predict - y_true), axis=-1))
    # dy = 2 * (y_predict - y_true) * relu_deriv(z) / y_true.shape[1]
    # loss = np.mean(np.mean(np.square(z - y_true), axis=-1))
    # dy = 2 * (z - y_true) / y_true.shape[1]
    # print('No active function define define for MSE')
    return loss, dy


def cross_entropy_loss(y_predict, y_true):
    """
    交叉熵损失函数
    :param y_predict: 预测值,shape (N,d)，N为批量样本数
    :param y_true: 真实值,shape(N,d)
    :return:
    """
    y_exp = np.exp(y_predict)
    y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
    loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 损失函数
    dy = y_probability - y_true
    return loss, dy


class MLP_Net:
    def __init__(self, sizes, loss_type='mse'):
        self.sizes = sizes
        self.mode = ['relu', 'sigmoid']
        self.num_layers = len(sizes)
        weights_scale = 0.01
        # building weights by [5000, 1024, 2] --> [[5000, 1024], [1024, 2]]
        self.weights = [np.random.randn(ch1, ch2) * weights_scale for ch1, ch2 in zip(sizes[:-1], sizes[1:])]
        # building biases by [5000, 1024, 2] --> [1024, 2]
        self.biases = [np.random.randn(1, ch) * weights_scale for ch in sizes[1:]]
        self.X = None
        self.Z = None

        self.loss_type = loss_type
        self.normalise = False
        self.dropout_X = None
        self.training = True

    def forward(self, x):
        self.X = [x]
        self.Z = []
        for layer_idx, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(x, w) + b
            if self.mode[layer_idx] == 'relu':
                x = relu(z)
            elif self.mode[layer_idx] == 'sigmoid':
                x = sigmoid(z)
            else:
                x = z
                print('active function is not defined for forward propagation')
            self.X.append(x)
            self.Z.append(z)
        return self.X[-1]  # y_predict

    def backward(self, y):   # y: ground truth
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        if self.loss_type == 'mse':
            loss, delta = mean_squared_loss(self.X[-1], y)
        else:
            loss, delta = cross_entropy_loss(self.X[-1], y)
        for l in range(self.num_layers - 2, -1, -1):
            x = self.X[l]
            db[l] = np.sum(delta, axis=0) / len(y)
            dw[l] = np.dot(x.T, delta) / len(y)
            if self.mode[l] == 'sigmoid':
                delta = np.dot(delta, self.weights[l].T) * sigmoid_deriv(self.Z[l - 1])
            elif self.mode[l] == 'relu':
                delta = np.dot(delta, self.weights[l].T) * relu_deriv(self.Z[l - 1])
            else:
                print('No active function define define for backward propagation')
        return dw, db

    def update_para(self, dw, db, lr, l1=0, l2=0):
        if l1 != 0:
            # L1范数正则化
            self.weights = [w - lr * (nabla + l1 * np.sign(w)) for w, nabla in zip(self.weights, dw)]
            self.biases = [b - lr * nabla for b, nabla in zip(self.biases, db)]
        elif l2 != 0:
            # L2范数正则化
            self.weights = [w - lr * (nabla + l2 * w) for w, nabla in zip(self.weights, dw)]
            self.biases = [b - lr * nabla for b, nabla in zip(self.biases, db)]
        else:
            # 不进行正则化
            self.weights = [w - lr * nabla for w, nabla in zip(self.weights, dw)]
            self.biases = [b - lr * nabla for b, nabla in zip(self.biases, db)]


def plot_trainning(order1, order2, img_name):
    '''
    画出训练过程的对比图
    :param order1: 第一种网络结构
    :param order2: 第二种网络结构
    :param img_name: 图片名称
    :return:
    '''
    with open(order1, 'rb') as f1, open(order2, 'rb') as f2:
        accs1 = pickle.load(f1)
        accs2 = pickle.load(f2)

    import matplotlib.pyplot as plt
    plt.figure()
    # x = [str(i) for i in range(1, len(accs1) + 1)]
    x = [i for i in range(1, len(accs1) + 1)]
    plt.plot(x, accs1, label=order1)
    plt.plot(x, accs2, label=order2)
    plt.legend()
    # plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(img_name)

def plot_single_training(order, img_name='best_acc.png'):
    '''
    画出最优参数下的训练过程
    :param order:
    :param img_name:
    :return:
    '''
    with open(order, 'rb') as f1:
        accs = pickle.load(f1)
    import matplotlib.pyplot as plt
    plt.figure()
    x = [i for i in range(1, len(accs) + 1)]
    plt.plot(x, accs)
    # plt.legend()
    # plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(img_name)


def train(net, train_samples, train_labels, test_samples, test_labels, epochs=1000, lr=0.1, l2=0, batch_size=1, l1=0, orders='first', gamma=1, step_size=0):
    lr0 = lr
    n_test = len(test_labels)
    n = train_samples.shape[0]
    accs = []
    for epoch in range(epochs):
        net.training = True
        for batch_index in range(0, n, batch_size):
            lower_range = batch_index
            upper_range = batch_index + batch_size
            if upper_range > n:
                upper_range = n
            train_x = train_samples[lower_range: upper_range, :]
            train_y = train_labels[lower_range: upper_range]
            net.forward(train_x)
            dw, db = net.backward(train_y)
            net.update_para(dw, db, lr, l1=l1, l2=l2)
    print(lr, end='\t')
    if step_size != 0:
        # 阶梯式衰减
        if (epoch + 1) % step_size == 0:
            lr *= gamma
    elif gamma != 1:
        # 指数衰减
        lr = math.pow(gamma, epoch) * lr0
    acc = evaluate(net, test_samples, test_labels)
    accs.append(acc / 10000.0)
    print('Epoch {0}: {1} / {2}'.format(epoch, acc / 10000.0, n_test))
    with open(orders, 'wb') as f:
        pickle.dump(accs, f)


def evaluate(net, test_images, test_labels):
    net.training = False
    result = []
    n = len(test_images)
    for batch_indx in range(0, n, 128):
        lower_range = batch_indx
        upper_range = batch_indx + 128
        if upper_range > n:
            upper_range = n
        test_x = test_images[lower_range: upper_range, :]
        result.extend(np.argmax(net.forward(test_x), axis=1))
    correct = sum(int(pred == y) for pred, y in zip(result, test_labels))
    return correct


def main():
    # data pre-load
    trainXDir = './data1forEx1to4/train1_icu_data.csv'
    trainYDir = './data1forEx1to4/train1_icu_label.csv'
    trainX, trainY = dataLoader(trainXDir, trainYDir)
    AtrainX, AtrainY = transftoArray(trainX, trainY)
    testXDir = './data1forEx1to4/test1_icu_data.csv'
    testYDir = './data1forEx1to4/test1_icu_label.csv'
    testX, testY = dataLoader(testXDir, testYDir)
    AtestX, AtestY = transftoArray(testX, testY)
    # network construct
    net = MLP_Net([108, 1024, 1], 'mse')
    orders1 = 'no_regular'
    train(net, AtrainX, AtrainY, AtestX, AtestY, epochs=100, orders=orders1, batch_size=64, lr=0.3, gamma=0.5, step_size=30)


if __name__ == '__main__':
    np.random.seed(1)
    main()


