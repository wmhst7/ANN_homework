# README

以注释的形式解释对文件的修改

```python
########################
# Additional Files
########################
# plot.py
# __pycache__
# data
# .DS_Store
# .idea

########################
# Filled Code
########################
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


########################
# References
########################

########################
# Other Modifications
########################
# _codes/run_mlp.py -> ../codes/run_mlp.py
# 7 + from plot import *
# 8 + import numpy as np
# 9 + import datetime
# 8 -
# 11 + starttime = datetime.datetime.now()
# 14 - model.add(Linear('fc1', 784, 10, 0.01))
# 17 + model.add(Linear('fc1', 784, 100, 0.01))
# 17 ?                                +
# 18 + model.add(Relu('relu1'))
# 19 + model.add(Linear('fc2', 100, 10, 0.01))
# 20 + # model.add(Relu('relu2'))
# 16 - loss = EuclideanLoss(name='loss')
# 22 + loss = SoftmaxCrossEntropyLoss(name='loss')
# 25 -     'learning_rate': 0.0,
# 25 ?                        ^
# 31 +     'learning_rate': 0.1,
# 31 ?                        ^
# 31 -     'test_epoch': 5
# 31 ?                   ^
# 37 +     'test_epoch': 1,
# 37 ?                   ^^
# 38 +     'plot_average_n': 100
# 41 + loss_list_train = []
# 42 + acc_list_train = []
# 43 + loss_list_test = []
# 44 + acc_list_test = []
# 37 -     train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
# 48 +     ll, al = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
# 48 ?    +++++++++
# 49 +     loss_list_train.extend(ll)
# 50 +     acc_list_train.extend(al)
# 39 -     if epoch % config['test_epoch'] == 0:
# 52 +     # if epoch % config['test_epoch'] == 0:
# 52 ?    ++
# 40 -         LOG_INFO('Testing @ %d epoch...' % (epoch))
# 40 ? ----
# 53 +     LOG_INFO('Testing @ %d epoch...' % (epoch))
# 41 -         test_net(model, loss, test_data, test_label, config['batch_size'])
# 41 ?       ^
# 54 +     ll, al = test_net(model, loss, test_data, test_label, config['batch_size'])
# 54 ?     +++ ++ ^
# 55 +     loss_list_test.extend(ll)
# 56 +     acc_list_test.extend(al)
# 57 +
# 58 + loss_list_train = average_list(loss_list_train, config['plot_average_n'])
# 59 + acc_list_train = average_list(acc_list_train, config['plot_average_n'])
# 60 +
# 61 + endtime = datetime.datetime.now()
# 62 +
# 63 + print(config)
# 64 + print('Final Train loss:', loss_list_train[-1], ", Train acc:", acc_list_train[-1])
# 65 + print('Final Test loss:', loss_list_test[-1], ", Test acc:", acc_list_test[-1])
# 66 + print('Total Time Used:', (endtime - starttime))
# 67 +
# 68 + plot_acc_loss(loss_list_train, acc_list_train, loss_list_test, acc_list_test)
# 69 +
# 70 +
# 71 +
# 72 +
# 73 +
# _codes/loss.py -> ../codes/loss.py
# 16 -         '''Your codes here'''
# 17 -         pass
# _codes/solve_net.py -> ../codes/solve_net.py
# 43 -             loss_list = []
# 43 +             # loss_list = []
# 43 ?            ++
# 44 -             acc_list = []
# 44 +             # acc_list = []
# 44 ?            ++
# 46 +     return np.ravel(loss_list), np.ravel(acc_list)
# 63 +     return np.ravel(np.mean(loss_list)), np.ravel(np.mean(acc_list))
# _codes/layers.py -> ../codes/layers.py
# 25 +
# 41 +
# 59 +
# 65 +     def f(self, x):
# 66 +         tanh = np.tanh(np.sqrt(2 / np.pi) * (x + 0.44715 * np.power(x, 3)))
# 67 +         res = 0.5 * (1.0 + tanh) * x
# 68 +         return res
# 69 +
# 86 +


```



