import numpy as np 
import matplotlib.pyplot as plt

class HingeLoss(object):
    def __init__(self, name, threshold=0.05):
        self.name = name

    def forward(self, input, target):
        # TODO START
        Delta = 5
        xt = np.max(np.where(target == 1, input, 0.), axis=1, keepdims=True)
        a = np.array(np.maximum(0., Delta - xt + input))
        h = np.where(input==1, 0., a)
        return np.sum(h) / input.shape[0]
        
        
        # TODO END

    def backward(self, input, target):
        # TODO START
        return (target - input) / input.shape[0]
        # TODO END

    def backward(self):
        # TODO START
        delta = 1e-5
        x = self._saved_tensor
        xx = x + delta
        res =  (self.f(xx) - self.fx) / delta
        return res
        # TODO END


if __name__ == "__main__":
    
    plt.plot(x, gx)
    plt.xlim((-8, 8))
    plt.ylim((-8, 8))
    plt.show()



