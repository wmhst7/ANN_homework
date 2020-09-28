import matplotlib.pyplot as plt 
import numpy as np


def average_list(ls, num):
    return [np.mean(ls[x: x + num]) for x in range(len(ls) - num + 1)]


def plot_acc_loss(loss_list_train, acc_list_train, loss_list_test, acc_list_test):
    fig, ax = plt.subplots(1, 2)

    par1 = ax[0].twinx()
    ax[0].set_xlabel("steps")
    ax[0].set_ylabel("loss")
    par1.set_ylabel("accuracy")
    p1, = ax[0].plot(range(len(loss_list_train)), loss_list_train, label="train loss", color="cornflowerblue")
    p2, = par1.plot(range(len(acc_list_train)), acc_list_train, label="train accuracy", color="coral")
    lines, labels = ax[0].get_legend_handles_labels()
    lines2, labels2 = par1.get_legend_handles_labels()
    ax[0].legend(lines + lines2, labels + labels2, loc=5)

    par2 = ax[1].twinx()
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    par2.set_ylabel("accuracy")
    p3, = ax[1].plot(range(len(loss_list_test)), loss_list_test, label="test loss", color="cornflowerblue")
    p4, = par2.plot(range(len(acc_list_test)), acc_list_test, label="test accuracy", color="coral")
    lines, labels = ax[1].get_legend_handles_labels()
    lines2, labels2 = par2.get_legend_handles_labels()
    ax[1].legend(lines + lines2, labels + labels2, loc=5)
    plt.tight_layout()
    plt.draw()
    plt.show()



