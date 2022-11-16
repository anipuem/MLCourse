import numpy as np
import random
import matplotlib.pyplot as plt
import math


def simple_smo(dataset, labels, C, max_iter):
    dataset = np.array(dataset)
    m, n = dataset.shape  # sample num，feature num
    labels = np.array(labels)
    # initialization
    lambds = np.zeros(m)  # one sample -- one λ
    b = 0
    it = 0

    while it < max_iter:
        pair_changed = 0  # check variance
        for i in range(m):
            λ_i, x_i, y_i = lambds[i], dataset[i], labels[i]
            fx_i = SVM_predict(x_i, lambds, dataset, labels, b)
            E_i = fx_i - y_i
            j = select_j(i, m)
            λ_j, x_j, y_j = lambds[j], dataset[j], labels[j]
            fx_j = SVM_predict(x_j, lambds, dataset, labels, b)
            E_j = fx_j - y_j
            K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(x_j, x_j), np.dot(x_i, x_j)
            eta = K_ii + K_jj - 2 * K_ij
            if eta <= 0:
                print('WARNING  eta <= 0')
                continue
            # update
            λ_i_old, λ_j_old = λ_i, λ_j  # 未更新前的参数
            λ_j_new = λ_j_old + y_j * (E_i - E_j) / eta
            # 对alpha进行修剪
            if y_i != y_j:
                L = max(0, λ_j_old - λ_i_old)
                H = min(C, C + λ_j_old - λ_i_old)
            else:
                L = max(0, λ_i_old + λ_j_old - C)
                H = min(C, λ_j_old + λ_i_old)
            λ_j_new = clip(λ_j_new, L, H)  # 根据上下界修剪
            λ_i_new = λ_i_old + y_i * y_j * (λ_j_old - λ_j_new)  # 根据公式反推另一个参数
            if abs(λ_j_new - λ_j_old) < 0.00001:  # 这个参数已经优化到最佳，换下一个
                # print('WARNING   alpha_j not moving enough')
                continue

            # update b
            lambds[i], lambds[j] = λ_i_new, λ_j_new
            b_i = -E_i - y_i * K_ii * (λ_i_new - λ_i_old) - y_j * K_ij * (λ_j_new - λ_j_old) + b
            b_j = -E_j - y_i * K_ij * (λ_i_new - λ_i_old) - y_j * K_jj * (λ_j_new - λ_j_old) + b
            if 0 < λ_i_new < C:
                b = b_i
            elif 0 < λ_j_new < C:
                b = b_j
            else:
                b = (b_i + b_j) / 2
            pair_changed += 1
            print('INFO   iteration:{}  i:{}  pair_changed:{}'.format(it, i, pair_changed))
        if pair_changed == 0:
            it += 1
        else:
            it = 0
        print('iteration number: {}'.format(it))
    return lambds, b


def SVM_predict(x, lambds, data, label, b):
    "SVM: y = w^Tx + b"
    res = 0
    for i in range(data.shape[0]):
        res += lambds[i] * label[i] * (data[i].dot(x.T))
    return res + b


def get_w(lambdas, dataset, labels):
    # get w
    w = 0
    for i in range(len(dataset)):
        w += lambdas[i] * y[i] * dataset[i]
    return w


def clip(alpha, L, H):
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha


def select_j(i, m):
    # random choose
    l = list(range(m))
    seq = l[: i] + l[i + 1:]
    return random.choice(seq)


def get_point():
    x_true = [[1, 2, 2], [2, 2, 1]]
    x_false = [[4, 4, 2]]
    x_all = np.array(x_true + x_false)
    y = [1] * len(x_true) + [-1] * len(x_false)
    return x_all, y, x_true, x_false


def plot(x_true, x_false, w, b):
    plot_x = np.arange(-1, 7)
    plot_y = -(w[0] * plot_x + b) / w[1]
    plt.scatter([x[0] for x in x_true], [x[1] for x in x_true], c='r', label='w1')
    plt.scatter([x[0] for x in x_false], [x[1] for x in x_false], c='b', label='w2')
    plt.plot(plot_x, plot_y, c='green')
    plt.xlim(-0.1, 2.1)
    plt.ylim(-0.1, 2.1)
    plt.legend()
    plt.grid(True)
    plt.plot()
    plt.show()


if __name__ == '__main__':
    x, y, x_true, x_false = get_point()
    lambdas, b = simple_smo(x, y, 10, 10)
    w = get_w(lambdas, x, y)
    print('-' * 10 + 'result' + '-' * 10)
    print('lambdas:{}\nw:{}\nb:{}'.format(lambdas, w, b))
    plot(x_true, x_false, w, b)
