from chainer import cuda
from chainer import optimizer


class EXNesterovAG(optimizer.GradientMethod):

    """
    !!!!Warning!!!!
    This "EXoptimizers" function is imenurok's original (2016/9/16, Chainer 1.14.0)
    Original "optimizers" is so good but I wondered the NesterovAG.
    I wrote original "originalNesterovAG", too.

    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['v'] = xp.zeros_like(param.data)

    def update_one_cpu(self, param, state):
        v = state['v']
        v *= self.momentum
        v -= self.lr * param.grad
        param.data += self.momentum * v
        param.data -= self.lr * param.grad

    def update_one_gpu(self, param, state):
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''v = v * momentum - lr * grad;
               param += momentum * momentum * v - (1 + momentum) * lr * grad;
               ''',
            'nesterov_ag')(param.grad, self.lr, self.momentum,
                           param.data, state['v'])



class originalNesterovAG(optimizer.GradientMethod):

    """Nesterov's Accelerated Gradient.

    Formulated as the linear combination coefficients of the velocity and
    gradient contributions at each iteration.

    See: http://arxiv.org/abs/1212.0901

    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['v'] = xp.zeros_like(param.data)

    def update_one_cpu(self, param, state):
        v = state['v']
        v *= self.momentum
        v -= self.lr * param.grad
        param.data += self.momentum * self.momentum * v
        param.data -= (1 + self.momentum) * self.lr * param.grad

    def update_one_gpu(self, param, state):
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''v = v * momentum - lr * grad;
               param += momentum * momentum * v - (1 + momentum) * lr * grad;
               ''',
            'nesterov_ag')(param.grad, self.lr, self.momentum,
                           param.data, state['v'])