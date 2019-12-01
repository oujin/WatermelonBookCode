from nn import FN, Sigmoid, Tanh, ReLU
import numpy as np
import matplotlib.pyplot as plt


class XOR(object):
    def __init__(self, batch_size, input_shape, hidden_shape, output_shape,
                 learning_rate, active_func):
        self.fn1 = FN(batch_size, input_shape, hidden_shape, learning_rate)
        self.active_func1 = active_func()
        self.fn2 = FN(batch_size, hidden_shape, output_shape, learning_rate)

    def forward(self, data):
        out = self.fn1.forward(data)
        out = self.active_func1.forward(out)
        out = self.fn2.forward(out)
        return out

    def backward(self, error):
        """BP"""
        delta = error
        delta = self.fn2.backward(delta)
        delta = self.active_func1.backward(delta)
        self.fn1.backward(delta)

    def predict(self, data):
        out = self.fn1.forward(data)
        out = self.active_func1.forward(out)
        print(out)
        out = self.fn2.forward(out)
        # out[out > .0] = 1.
        # out[out < .0] = 0.
        out[out >= .5] = 1.
        out[out < .5] = 0.
        return out


def norm(data):
    return np.sum(data ** 2)


if __name__ == "__main__":
    # experiment 1.1
    # xor problem
    xor = XOR(4, 2, 2, 1, 0.1, Sigmoid)
    data = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    res = np.array([[1], [1], [0], [0]])
    loss_s = []
    # 5 experiments
    for j in range(5):
        for i in range(100000):
            out = xor.forward(data)
            xor.backward(out - res)
        out = xor.predict(data)
        print(out)

    # experiment 1.2
    # testing Sigmoid loss
    xor = XOR(4, 2, 2, 1, 0.1, Sigmoid)
    data = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    res = np.array([[1], [1], [0], [0]])
    loss_s = []
    for i in range(10000):
        out = xor.forward(data)
        xor.backward(out - res)
        if i % 100 == 0:
            loss_s.append([i, norm(out - res)])
    loss_s = np.asarray(loss_s)
    # testing Relu loss
    xor = XOR(4, 2, 2, 1, 0.1, ReLU)
    loss_r = []
    for i in range(10000):
        out = xor.forward(data)
        xor.backward(out - res)
        if i % 100 == 0:
            loss_r.append([i, norm(out - res)])
    loss_r = np.asarray(loss_r)
    # testing Tanh loss
    xor = XOR(4, 2, 2, 1, 0.1, Tanh)
    loss_t = []
    for i in range(10000):
        out = xor.forward(data)
        xor.backward(out - res)
        if i % 100 == 0:
            loss_t.append([i, norm(out - res)])
    loss_t = np.asarray(loss_t)
    # draw the figure
    plt.figure()
    plt.plot(loss_s[:, 0], loss_s[:, 1], label="sigmoid")
    plt.plot(loss_r[:, 0], loss_r[:, 1], label="relu")
    plt.plot(loss_t[:, 0], loss_t[:, 1], label="tanh")
    plt.xlabel("training times")
    plt.ylabel("training loss")
    plt.legend()
    plt.show()
