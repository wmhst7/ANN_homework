from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Gelu
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
from plot import *
import numpy as np
import datetime

starttime = datetime.datetime.now()
train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
# 1
# model = Network()
# model.add(Linear('fc1', 784, 256, 0.01))
# model.add(Relu('relu1'))
# model.add(Linear('fc2', 256, 10, 0.01))
# 2
model = Network()
model.add(Linear('fc1', 784, 500, 0.01))
model.add(Relu('relu1'))
model.add(Linear('fc2', 500, 100, 0.01))
model.add(Relu('relu2'))
model.add(Linear('fc3', 100, 10, 0.01))

loss = EuclideanLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.1,
    'weight_decay': 0.0,
    'momentum': 0.0,
    'batch_size': 100,
    'max_epoch': 60,
    'disp_freq': 500,
    'test_epoch': 1,
    'plot_average_n': 100
}

loss_list_train = []
acc_list_train = []
loss_list_test = []
acc_list_test = []

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    ll, al = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    loss_list_train.extend(ll)
    acc_list_train.extend(al)

    # if epoch % config['test_epoch'] == 0:
    LOG_INFO('Testing @ %d epoch...' % (epoch))
    ll, al = test_net(model, loss, test_data, test_label, config['batch_size'])
    loss_list_test.extend(ll)
    acc_list_test.extend(al)

loss_list_train = average_list(loss_list_train, config['plot_average_n'])
acc_list_train = average_list(acc_list_train, config['plot_average_n'])

endtime = datetime.datetime.now()

print("INFO\n", config)
print('Final Train loss:', loss_list_train[-1], ", Train acc:", acc_list_train[-1])
print('Final Test loss:', loss_list_test[-1], ", Test acc:", acc_list_test[-1])
print('Total Time Used:', (endtime - starttime))

np.save('res/loss_list_train_2_euc.npy', loss_list_train)
np.save('res/acc_list_train_2_euc.npy', acc_list_train)
np.save('res/loss_list_test_2_euc.npy', loss_list_test)
np.save('res/acc_list_test_2_euc.npy', acc_list_test)

plot_acc_loss(loss_list_train, acc_list_train, loss_list_test, acc_list_test)





