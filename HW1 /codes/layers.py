import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        return np.maximum(input, 0)
        # TODO END

    def backward(self, grad_output):
        # TODO START
        grad_output[self._saved_tensor <= 0] = 0
        return grad_output
        # TODO END


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        # TODO START
        res = 1.0 / (1.0 + np.exp(-input))
        self._saved_for_backward(res)
        return res
        # TODO END

    def backward(self, grad_output):
        # TODO START
        res = grad_output * self._saved_tensor * (1.0 - self._saved_tensor)
        return res
        # TODO END


class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)

    def f(self, x):
        tanh = np.tanh(np.sqrt(2 / np.pi) * (x + 0.44715 * np.power(x, 3)))
        res = 0.5 * (1.0 + tanh) * x
        return res

    def forward(self, input):
        # TODO START
        res = self.f(input)
        self._saved_for_backward(input)
        self.fx = res
        return res
        # TODO END

    def backward(self, grad_output):
        # TODO START
        delta = 1e-5
        x = self._saved_tensor
        xx = x + delta
        res = grad_output * (self.f(xx) - self.fx) / delta
        return res
        # TODO END


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        # print("Linear Forward: Shape of input: ", input.shape)
        self._saved_for_backward(input)
        self._saved_tensor = input
        return input.dot(self.W) + self.b  # Wx + b
        # TODO END

    def backward(self, grad_output):
        # TODO START
        self.grad_W = self._saved_tensor.T.dot(grad_output)  # W
        self.grad_b = grad_output.sum(axis=0)  # b
        return grad_output.dot(self.W.T)  # W^T * grad_output
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
