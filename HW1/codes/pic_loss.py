import matplotlib.pyplot as plt
import numpy as np


def average_list(ls, num):
    return [np.mean(ls[x: x + num]) for x in range(len(ls) - num + 1)]


if __name__ == "__main__":

    acc_list_test_1_softmax = np.load("./res/acc_list_test_1_softmax.npy")
    acc_list_train_1_softmax = np.load("./res/acc_list_train_1_softmax.npy")
    loss_list_test_1_softmax = np.load("./res/loss_list_test_1_softmax.npy")
    loss_list_train_1_softmax = np.load("./res/loss_list_train_1_softmax.npy")

    acc_list_test_1_euc = np.load("./res/acc_list_test_1_euc.npy")
    acc_list_train_1_euc = np.load("./res/acc_list_train_1_euc.npy")
    loss_list_test_1_euc = np.load("./res/loss_list_test_1_euc.npy")
    loss_list_train_1_euc = np.load("./res/loss_list_train_1_euc.npy")

    acc_list_test_1_hinge = np.load("./res/acc_list_test_1_hinge.npy")
    acc_list_train_1_hinge = np.load("./res/acc_list_train_1_hinge.npy")
    loss_list_test_1_hinge = np.load("./res/loss_list_test_1_hinge.npy")
    loss_list_train_1_hinge = np.load("./res/loss_list_train_1_hinge.npy")

    # plot Train
    fig, ax = plt.subplots(1, 2)
    # loss
    ax[0].set_xlabel("steps")
    ax[0].set_ylabel("loss")
    ax[0].plot(range(len(loss_list_train_1_softmax)), loss_list_train_1_softmax,
               label="train loss with SoftmaxEntropyLoss", color="royalblue")
    ax[0].plot(range(len(loss_list_train_1_euc)), loss_list_train_1_euc,
               label="train loss with EuclideanLoss",
               color="tomato")
    ax[0].plot(range(len(loss_list_train_1_hinge)), loss_list_train_1_hinge,
               label="train loss with HingeLoss",
               color="y")
    ax[0].legend(loc=5)
    # acc
    ax[1].set_xlabel("steps")
    ax[1].set_ylabel("accuracy")
    ax[1].plot(range(len(acc_list_train_1_softmax)), acc_list_train_1_softmax,
               label="train acc with SoftmaxEntropyLoss", color="royalblue")
    ax[1].plot(range(len(acc_list_train_1_euc)), acc_list_train_1_euc,
               label="train acc with EuclideanLoss", color="tomato")
    ax[1].plot(range(len(acc_list_train_1_hinge)), acc_list_train_1_hinge,
               label="train acc with HingeLoss", color="y")
    ax[1].legend(loc=5)
    # final
    plt.tight_layout()
    plt.draw()
    plt.show()

    # # plot Test
    # fig, ax = plt.subplots(1, 2)
    # # loss
    # ax[0].set_xlabel("epochs")
    # ax[0].set_ylabel("loss")
    # ax[0].plot(range(len(loss_list_test_1_softmax)), loss_list_test_1_softmax, label="test loss with SoftmaxEntropyLoss",
    #            color="royalblue")
    # ax[0].plot(range(len(loss_list_test_1_euc)), loss_list_test_1_euc, label="test loss with EuclideanLoss",
    #            color="tomato")
    # ax[0].plot(range(len(loss_list_test_1_hinge)), loss_list_test_1_hinge, label="test loss with HingeLoss", color="y")
    # ax[0].legend(loc=5)
    # # acc
    # ax[1].set_xlabel("epochs")
    # ax[1].set_ylabel("accuracy")
    # ax[1].plot(range(len(acc_list_test_1_softmax)), acc_list_test_1_softmax, label="test acc with SoftmaxEntropyLoss", color="royalblue")
    # ax[1].plot(range(len(acc_list_test_1_euc)), acc_list_test_1_euc, label="test acc with EuclideanLoss",
    #            color="tomato")
    # ax[1].plot(range(len(acc_list_test_1_hinge)), acc_list_test_1_hinge, label="test acc with HingeLoss", color="y")
    # ax[1].legend(loc=5)
    # # final
    # plt.tight_layout()
    # plt.draw()
    # plt.show()

