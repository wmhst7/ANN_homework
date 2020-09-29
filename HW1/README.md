# README

在注释中解释对文件的修改

in `summary.txt` :

```python
########################
# Additional Files
########################
# res
# pic_loss.py
# .DS_Store
# plot.py
# .idea
# pic_activity.py
# data
# __pycache__

########################
# Filled Code
########################
# ../codes/layers.py:1
        self._saved_for_backward(input)
        return np.maximum(input, 0)

# ../codes/layers.py:2
        grad_output[self._saved_tensor <= 0] = 0
        return grad_output

# ../codes/layers.py:3
        res = 1.0 / (1.0 + np.exp(-input))
        self._saved_for_backward(res)
        return res

# ../codes/layers.py:4
        res = grad_output * self._saved_tensor * (1.0 - self._saved_tensor)
        return res

# ../codes/layers.py:5
        res = self.f(input)
        self._saved_for_backward(input)
        self.fx = res
        return res

# ../codes/layers.py:6
        delta = 1e-5
        x = self._saved_tensor
        xx = x + delta
        res = grad_output * (self.f(xx) - self.fx) / delta
        return res

# ../codes/layers.py:7
        # print("Linear Forward: Shape of input: ", input.shape)
        self._saved_for_backward(input)
        self._saved_tensor = input
        return input.dot(self.W) + self.b  # Wx + b

# ../codes/layers.py:8
        self.grad_W = self._saved_tensor.T.dot(grad_output)  # W
        self.grad_b = grad_output.sum(axis=0)  # b
        return grad_output.dot(self.W.T)  # W^T * grad_output

# ../codes/loss.py:1
        return np.sum((target - input) ** 2) / (2.0 * input.shape[0])

# ../codes/loss.py:2
        inexp = np.exp(input)
        pk = inexp / np.sum(inexp, axis=1, keepdims=True)
        Ek = -np.sum(target * np.log(pk)) / input.shape[0]
        self.softmax = pk
        return Ek

# ../codes/loss.py:3
        pk = self.softmax
        return (pk / np.sum(target, axis=1, keepdims=True) - target) / input.shape[0]

# ../codes/loss.py:4
        Delta = 5
        x = np.max(np.where(target == 1, input, 0.), axis=1, keepdims=True)
        a = np.array(np.maximum(0., Delta - x + input))
        b = np.where(target == 1, 0., a)
        res = np.sum(b) / input.shape[0]
        self.b = b
        return res

# ../codes/loss.py:5
        b = self.b
        c = np.where(b > 0., 1., 0.)
        m = np.zeros(input.shape) - np.sum(c, axis=1, keepdims=True)
        res = np.where(target == 1, m, c) / input.shape[0]
        return res


########################
# References
########################

########################
# Other Modifications
########################
# _codes/solve_net.py -> ../codes/solve_net.py 记录loss与acc用于绘图
# 43 -             loss_list = []
# 43 +             # loss_list = []
# 43 ?            ++
# 44 -             acc_list = []
# 44 +             # acc_list = []
# 44 ?            ++
# 46 +     return np.ravel(loss_list), np.ravel(acc_list)
# 63 +     return np.ravel(np.mean(loss_list)), np.ravel(np.mean(acc_list))
# _codes/layers.py -> ../codes/layers.py 为了便于计算gelu设置函数
# 25 +
# 41 +
# 59 +
# 65 +     def f(self, x):
# 66 +         tanh = np.tanh(np.sqrt(2 / np.pi) * (x + 0.44715 * np.power(x, 3)))
# 67 +         res = 0.5 * (1.0 + tanh) * x
# 68 +         return res
# 69 +
# 86 +
# _codes/run_mlp.py -> ../codes/run_mlp.py 
# 7 + from plot import *
# 8 + import numpy as np
# 9 + import datetime
# 8 -
# 11 + starttime = datetime.datetime.now() 记录时间
# 16 + # 1
# 17 + # model = Network() 不同的神经网络
# 18 + # model.add(Linear('fc1', 784, 256, 0.01))
# 19 + # model.add(Relu('relu1'))
# 20 + # model.add(Linear('fc2', 256, 10, 0.01))
# 21 + # 2
# 14 - model.add(Linear('fc1', 784, 10, 0.01))
# 14 ?                              ^
# 23 + model.add(Linear('fc1', 784, 500, 0.01))
# 23 ?                              ^^
# 24 + model.add(Relu('relu1'))
# 25 + model.add(Linear('fc2', 500, 100, 0.01))
# 26 + model.add(Relu('relu2'))
# 27 + model.add(Linear('fc3', 100, 10, 0.01))
# 16 - loss = EuclideanLoss(name='loss')
# 29 + loss = SoftmaxCrossEntropyLoss(name='loss')
# 25 -     'learning_rate': 0.0, 修改参数
# 25 ?                        ^
# 38 +     'learning_rate': 0.1,
# 38 ?                        ^
# 29 -     'max_epoch': 100,
# 29 ?                  ^^
# 42 +     'max_epoch': 60,
# 42 ?                  ^
# 30 -     'disp_freq': 50,
# 43 +     'disp_freq': 500,
# 43 ?                    +
# 31 -     'test_epoch': 5
# 31 ?                   ^
# 44 +     'test_epoch': 1,
# 44 ?                   ^^
# 45 +     'plot_average_n': 100
# 48 + loss_list_train = []
# 49 + acc_list_train = []
# 50 + loss_list_test = []
# 51 + acc_list_test = []
# 37 -     train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
# 55 +     ll, al = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
# 55 ?    +++++++++
# 56 +     loss_list_train.extend(ll) 记录loss、acc用于绘图
# 57 +     acc_list_train.extend(al)
# 39 -     if epoch % config['test_epoch'] == 0:
# 59 +     # if epoch % config['test_epoch'] == 0:
# 59 ?    ++
# 40 -         LOG_INFO('Testing @ %d epoch...' % (epoch))
# 40 ? ----
# 60 +     LOG_INFO('Testing @ %d epoch...' % (epoch))
# 41 -         test_net(model, loss, test_data, test_label, config['batch_size'])
# 41 ?       ^
# 61 +     ll, al = test_net(model, loss, test_data, test_label, config['batch_size'])
# 61 ?     +++ ++ ^
# 62 +     loss_list_test.extend(ll)
# 63 +     acc_list_test.extend(al)
# 64 +
# 65 + loss_list_train = average_list(loss_list_train, config['plot_average_n'])
# 66 + acc_list_train = average_list(acc_list_train, config['plot_average_n'])
# 67 +
# 68 + endtime = datetime.datetime.now()
# 69 +
# 70 + print("INFO\n", config)
# 71 + print('Final Train loss:', loss_list_train[-1], ", Train acc:", acc_list_train[-1])
# 72 + print('Final Test loss:', loss_list_test[-1], ", Test acc:", acc_list_test[-1])
# 73 + print('Total Time Used:', (endtime - starttime))
# 74 +
# 75 + np.save('res/loss_list_train_2_hinge.npy', loss_list_train) 保存数据用于绘图
# 76 + np.save('res/acc_list_train_2_hinge.npy', acc_list_train)
# 77 + np.save('res/loss_list_test_2_hinge.npy', loss_list_test)
# 78 + np.save('res/acc_list_test_2_hinge.npy', acc_list_test)
# 79 +
# 80 + plot_acc_loss(loss_list_train, acc_list_train, loss_list_test, acc_list_test)
# 81 +
# 82 +
# 83 +
# 84 +
# 85 +
# _codes/loss.py -> ../codes/loss.py
# 16 -         '''Your codes here'''
# 17 -         pass


```

此外，`code/plot.py`用于训练完后绘图。

`code/pic_activity.py`、`code/loss.py`用于统计绘图。

