import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rdm

if __name__ == "__main__":
    # 1
    data = rdm.normal(10, 9, (300, 2))
    data = np.vstack((data, rdm.normal(50, 7, (300, 2))))
    data = np.vstack((data, rdm.normal(35, 5, (300, 2))))
    tags = np.vstack((np.ones((300, 1)),
                      np.ones((300, 1)) * 2,
                      np.ones((300, 1)) * 3))
    index = np.arange(900)
    data = np.hstack((data, tags))
    for i in range(10):
        points = data[data[:, -1] == i]
        plt.scatter(points[:, 0], points[:, 1])
    plt.show()
    np.save("data/1.npy", data)

    # 2
    data = rdm.uniform([10, 12], [50, 20], (300, 2))
    data = np.vstack((data, rdm.uniform([4, 0], [49, 10], (300, 2))))
    data = np.vstack((data, rdm.uniform([12, 30], [55, 22], (300, 2))))
    tags = np.vstack((np.ones((300, 1)),
                      np.ones((300, 1)) * 2,
                      np.ones((300, 1)) * 3))
    index = np.arange(900)
    data = np.hstack((data, tags))
    for i in range(10):
        points = data[data[:, -1] == i]
        plt.scatter(points[:, 0], points[:, 1])
    plt.show()
    np.save("data/2.npy", data)

    # 3
    data = rdm.uniform([10, 10], [11, 50], (300, 2))
    lm = np.linspace(0, 2 * np.pi, 300)
    data[:, 0] = data[:, 0] + lm * 5
    data[:, -1] = data[:, -1] + np.sin(lm) * 50
    data = np.vstack((data, rdm.uniform([27, 50], [27, 90], (300, 2))))
    lm = np.linspace(0, 2 * np.pi, 300)
    data[300:, 0] = data[300:, 0] + lm * 5
    data[300:, -1] = data[300:, -1] + np.sin(lm) * 50
    data = np.vstack((data, rdm.logistic(35, 1, (300, 2))))
    data[600:, -1] = data[600:, -1] * 4 - 90
    data = np.vstack((data, rdm.uniform([15, -25], [20, 50], (300, 2))))
    data = np.vstack((data, rdm.normal((50, 80), (2, 8), (300, 2))))
    data[:, -1] = data[:, -1] / 3 + 20
    tags = np.vstack((np.ones((300, 1)),
                      np.ones((300, 1)) * 2,
                      np.ones((300, 1)) * 3,
                      np.ones((300, 1)) * 4,
                      np.ones((300, 1)) * 5))
    index = np.arange(1500)
    data = np.hstack((data, tags))
    for i in range(10):
        points = data[data[:, -1] == i]
        plt.scatter(points[:, 0], points[:, 1])
    plt.xlim(0, 70)
    plt.ylim(0, 70)
    plt.show()
    np.save("data/3.npy", data)
