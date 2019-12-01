from nn import FN, Sigmoid
import numpy as np
import matplotlib.pyplot as plt


class AutoEncoder(object):
    def __init__(self, batch_size, input_shape, hidden_shape,
                 learning_rate, active_func):
        self.fn1 = FN(batch_size, input_shape, hidden_shape, learning_rate)
        self.active_func1 = active_func()
        self.fn2 = FN(batch_size, hidden_shape, input_shape, learning_rate)
        self.active_func2 = active_func()

    def forward(self, data):
        out = self.fn1.forward(data)
        out = self.active_func1.forward(out)
        out = self.fn2.forward(out)
        out = self.active_func2.forward(out)
        return out

    def backward(self, error):
        delta = self.active_func2.backward(error)
        delta = self.fn2.backward(delta)
        delta = self.active_func1.backward(delta)
        self.fn1.backward(delta)


class AutoEncoder1(object):
    def __init__(self, batch_size, input_shape, hidden_shape,
                 learning_rate, active_func):
        self.fn1 = FN(batch_size, input_shape, hidden_shape, learning_rate)
        self.active_func1 = active_func()
        self.fn2 = FN(batch_size, hidden_shape, input_shape, learning_rate)
        self.active_func2 = active_func()
        # share the weights
        self.fn2.weights[:-1, :] = self.fn1.weights[:-1, :].T

    def forward(self, data):
        # forward
        out = self.fn1.forward(data)
        out = self.active_func1.forward(out)
        out = self.fn2.forward(out)
        out = self.active_func2.forward(out)
        return out

    def backward(self, error):
        """BP"""
        delta = self.active_func2.backward(error)
        delta = self.fn2.backward(delta)
        delta = self.active_func1.backward(delta)
        self.fn1.backward(delta)
        # the shared weights
        # self.fn1.weights += (delta_1 + delta_2.T)
        weights = (self.fn1.weights[:-1, :] + self.fn2.weights[:-1, :].T) / 2
        self.fn1.weights[:-1, :] = weights
        self.fn2.weights[:-1, :] = weights.T


def norm(data):
    return np.sum(data ** 2)


if __name__ == "__main__":
    # # experiment 2.1
    # auto_encoder = AutoEncoder(8, 8, 3, 1, Sigmoid)
    # data = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 1, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 1, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 1, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 1, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 1]])
    # # train
    # for i in range(100000):
    #     res = auto_encoder.forward(data)
    #     auto_encoder.backward(res - data)
    # # test
    # res = auto_encoder.fn1.forward(data)
    # res = auto_encoder.active_func1.forward(res)
    # res[res > .5] = 1.
    # res[res < .5] = 0.
    # print(res)
    # res = auto_encoder.forward(data)
    # res[res > .5] = 1.
    # res[res < .5] = 0.
    # print(res)

    # # experiment 2.2 weights-shared autoencoder
    # auto_encoder = AutoEncoder1(8, 8, 3, 1, Sigmoid)
    # data = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 1, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 1, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 1, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 1, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 1, 0],
    #                  [0, 0, 0, 0, 0, 0, 0, 1]])
    # # train
    # for i in range(100000):
    #     res = auto_encoder.forward(data)
    #     auto_encoder.backward(res - data)
    # # test
    # res = auto_encoder.fn1.forward(data)
    # res = auto_encoder.active_func1.forward(res)
    # res[res > .5] = 1.
    # res[res < .5] = 0.
    # print(res)
    # res = auto_encoder.forward(data)
    # res[res > .5] = 1.
    # res[res < .5] = 0.
    # print(res)

    # experiment 2.3 testing the loss
    auto_encoder = AutoEncoder(8, 8, 3, 1, Sigmoid)
    auto_encoder1 = AutoEncoder1(8, 8, 3, 1, Sigmoid)
    data = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1]])
    # train
    loss = []
    loss_s = []
    # train weights-unshared net
    for i in range(10000):
        res = auto_encoder.forward(data)
        auto_encoder.backward(res - data)
        if i % 100 == 0:
            loss.append([i, norm(res - data)])
    # train weights-shared net
    for i in range(10000):
        res = auto_encoder1.forward(data)
        auto_encoder1.backward(res - data)
        if i % 100 == 0:
            loss_s.append([i, norm(res - data)])
    loss = np.asarray(loss)
    loss_s = np.asarray(loss_s)
    # draw the figure
    plt.figure()
    plt.plot(loss[:, 0], loss[:, 1], label="weights-unshared")
    plt.plot(loss_s[:, 0], loss_s[:, 1], label="weights-shared")
    plt.xlabel("training times")
    plt.ylabel("training loss")
    plt.legend()
    plt.show()
