import matplotlib.pyplot as plt
import numpy as np


def average_list(ls, num):
    return [np.mean(ls[x: x + num]) for x in range(len(ls) - num + 1)]


if __name__ == "__main__":

    acc_list_test_2_relu = np.load("./res/acc_list_test_2_relu.npy")
    acc_list_train_2_relu = np.load("./res/acc_list_train_2_relu.npy")
    loss_list_test_2_relu = np.load("./res/loss_list_test_2_relu.npy")
    loss_list_train_2_relu = np.load("./res/loss_list_train_2_relu.npy")

    acc_list_test_2_gelu = np.load("./res/acc_list_test_2_gelu.npy")
    acc_list_train_2_gelu = np.load("./res/acc_list_train_2_gelu.npy")
    loss_list_test_2_gelu = np.load("./res/loss_list_test_2_gelu.npy")
    loss_list_train_2_gelu = np.load("./res/loss_list_train_2_gelu.npy")

    acc_list_test_2_sigmoid = np.load("./res/acc_list_test_2_sigmoid.npy")
    acc_list_train_2_sigmoid = np.load("./res/acc_list_train_2_sigmoid.npy")
    loss_list_test_2_sigmoid = np.load("./res/loss_list_test_2_sigmoid.npy")
    loss_list_train_2_sigmoid = np.load("./res/loss_list_train_2_sigmoid.npy")

    # # plot Train MLP1 Activity
    # fig, ax = plt.subplots(1, 2)
    # # loss
    # ax[0].set_xlabel("steps")
    # ax[0].set_ylabel("loss")
    # ax[0].plot(range(len(loss_list_train_2_relu)), loss_list_train_2_relu, label="train loss with Relu", color="royalblue")
    # ax[0].plot(range(len(loss_list_train_2_sigmoid)), loss_list_train_2_sigmoid, label="train loss with Sigmoid", color="tomato")
    # ax[0].plot(range(len(loss_list_train_2_gelu)), loss_list_train_2_gelu, label="train loss with Gelu", color="y")
    # ax[0].legend(loc=5)
    # # acc
    # ax[1].set_xlabel("steps")
    # ax[1].set_ylabel("accuracy")
    # ax[1].plot(range(len(acc_list_train_2_relu)), acc_list_train_2_relu, label="train acc with Relu", color="royalblue")
    # ax[1].plot(range(len(acc_list_train_2_sigmoid)), acc_list_train_2_sigmoid, label="train acc with Sigmoid", color="tomato")
    # ax[1].plot(range(len(acc_list_train_2_gelu)), acc_list_train_2_gelu, label="train acc with Gelu", color="y")
    # ax[1].legend(loc=5)
    # # final
    # plt.tight_layout()
    # plt.draw()
    # plt.show()

    # plot Test MLP1 Activity
    fig, ax = plt.subplots(1, 2)
    # loss
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")
    ax[0].plot(range(len(loss_list_test_2_relu)), loss_list_test_2_relu, label="test loss with Relu",
               color="royalblue")
    ax[0].plot(range(len(loss_list_test_2_sigmoid)), loss_list_test_2_sigmoid, label="test loss with Sigmoid",
               color="tomato")
    ax[0].plot(range(len(loss_list_test_2_gelu)), loss_list_test_2_gelu, label="test loss with Gelu", color="y")
    ax[0].legend(loc=5)
    # acc
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("accuracy")
    ax[1].plot(range(len(acc_list_test_2_relu)), acc_list_test_2_relu, label="test acc with Relu", color="royalblue")
    ax[1].plot(range(len(acc_list_test_2_sigmoid)), acc_list_test_2_sigmoid, label="test acc with Sigmoid",
               color="tomato")
    ax[1].plot(range(len(acc_list_test_2_gelu)), acc_list_test_2_gelu, label="test acc with Gelu", color="y")
    ax[1].legend(loc=5)
    # final
    plt.tight_layout()
    plt.draw()
    plt.show()

    # plot Train MLP1 Loss
