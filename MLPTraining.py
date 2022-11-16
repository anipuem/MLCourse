import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from loadData import dataLoader
from loadData import transftoArray
from MLP import MLP


def draw_plot(Full_Epoch, train_loss_list, train_acc_list, val_loss_list, val_acc_list):
    x = range(0, Full_Epoch)
    plt.figure(figsize=(13, 13))
    plt.subplot(2, 1, 1)
    plt.plot(x, train_loss_list, color="blue", label="train_loss_list Line", linewidth=2)
    plt.plot(x, val_loss_list, color="orange", label="test_loss_list Line", linewidth=2)
    plt.title("Loss_curve", fontsize=20)
    plt.xlabel(xlabel="Epochs", fontsize=15)
    plt.ylabel(ylabel="Loss", fontsize=15)
    plt.legend()
    plt.subplot(2, 1, 2)
    aa = np.ones((len(train_acc_list),))-train_acc_list
    bb = np.ones((len(val_acc_list),)) - val_acc_list
    plt.plot(x, aa, color="blue", label="train_acc_list Line", linewidth=2)
    plt.plot(x, bb, color="orange", label="test_acc_list Line", linewidth=2)
    plt.title("Acc_curve", fontsize=20)
    plt.xlabel(xlabel="Epochs", fontsize=15)
    plt.ylabel(ylabel="Accuracy", fontsize=15)
    plt.legend()
    plt.savefig("Loss&acc.jpg")


def train_one_epoch(model, loss_func, epoch, epoch_size, epoch_size_val, gen, gen_val, Full_Epoch):
    train_loss = 0
    val_loss = 0
    total_loss = 0
    total_val_loss = 0

    print(f"\n Start training epoch{epoch + 1}")

    with tqdm(total=epoch_size, desc=f'Epoch{epoch + 1}/{Full_Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            # 一次读取的样本数是Batch_Size个
            samplesT, labelsT = batch
            optimizer.zero_grad()
            output = model(samplesT)
            loss = loss_func(output, labelsT)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_loss = total_loss / (iteration + 1)
            train_acc = mean_absolute_error(labelsT.detach().numpy(), output.detach().numpy())

            pbar.set_postfix(**{"total_loss": train_loss,
                                "learning_rate:": 1e-4,
                                "Acc": train_acc})
            pbar.update(1)  # 更新进度条

    print(f"\n Finish epoch{epoch + 1} parameter update")
    print(f"Start to evaluate epoch{epoch + 1}")
    with tqdm(total=epoch_size_val, desc=f'Epoch{epoch + 1}/{Full_Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_size_val:
                break
            samples, labels = batch
            optimizer.zero_grad()
            output = model(samples)
            loss = loss_func(output, labels)

            total_val_loss += loss.item()
            val_loss = total_val_loss / (iteration + 1)
            val_acc = mean_absolute_error(labels.detach().numpy(), output.detach().numpy())

            pbar.set_postfix(**{"val_loss": val_loss,
                                "Acc": val_acc})
            pbar.update(1)  # 更新进度条
    if epoch + 1 == Full_Epoch:
        torch.save(model.state_dict(),
                   'weights/mlp_weights-epoch%d-train_loss%.2f-test_loss%.2f.pkl' % (
                   (epoch + 1), train_loss, val_loss / (iteration + 1)))

    return train_loss, train_acc, val_loss, val_acc


if __name__ == "__main__":
    Full_Epoch = 300
    Batch_size = 64
    lr = 1e-3
    loss_and_acc_curve = True

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    # data pre-load
    trainXDir = './data1forEx1to4/train1_icu_data.csv'
    trainYDir = './data1forEx1to4/train1_icu_label.csv'
    trainX, trainY = dataLoader(trainXDir, trainYDir)
    x_train_s, y_train = transftoArray(trainX, trainY)
    testXDir = './data1forEx1to4/test1_icu_data.csv'
    testYDir = './data1forEx1to4/test1_icu_label.csv'
    testX, testY = dataLoader(testXDir, testYDir)
    x_test_s, y_test = transftoArray(testX, testY)
    x_train_f = torch.from_numpy(x_train_s.astype(np.float32))
    x_test_f = torch.from_numpy(x_test_s.astype(np.float32))
    y_train_f = torch.from_numpy(y_train.astype(np.float32))
    y_test_f = torch.from_numpy(y_test.astype(np.float32))

    train_dataset = TensorDataset(x_train_f, y_train_f)
    test_dataset = TensorDataset(x_test_f, y_test_f)
    gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=0, shuffle=True)
    gen_val = DataLoader(test_dataset, batch_size=Batch_size, num_workers=0, shuffle=False)

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr)
    loss_func = nn.MSELoss()
    epoch_size = y_train_f.size(0) // Batch_size  # epoch_size time to update parameters
    epoch_size_val = y_test_f.size(0) // Batch_size

    for epoch in range(0, Full_Epoch):
        train_loss, train_acc, val_loss, val_acc = train_one_epoch(model, loss_func, epoch, epoch_size, epoch_size_val,
                                                                   gen, gen_val, Full_Epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

    if loss_and_acc_curve:
        draw_plot(Full_Epoch, train_loss_list, train_acc_list, val_loss_list, val_acc_list)

