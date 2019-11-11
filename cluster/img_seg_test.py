import cv2
import matplotlib.pyplot as plt
import numpy as np

from kmeans import k_means
from gmm import gmm
from dbscan import dbscan
from distance import euclidean_distance


def dist(x, y):
    pos_d = euclidean_distance(x[:, :2], y[:2])
    pix_d = euclidean_distance(x[:, 2:], y[2:])
    d = np.sqrt(pos_d ** 2 + 5 * pix_d ** 2)
    return d


if __name__ == "__main__":
    # color image
    img = cv2.imread('data/0.jpg',)[:, :, ::-1]
    plt.figure(1)
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(img)
    plt.subplot(122)
    c = np.loadtxt('data/0.txt')
    plt.axis('off')
    plt.imshow(c)
    plt.show()
    pos = np.zeros((img.shape[0], img.shape[1], 2))
    for i in range(pos.shape[0]):
        for j in range(pos.shape[1]):
            pos[i, j, :] = [i, j]
    pos = pos.reshape((-1, 2))
    data = np.hstack((pos, np.reshape(img, (-1, 3))))
    # k_means
    res = k_means(data, 3, iter_times=20, dist_func=dist)
    tag = res[:, -1]
    tag = np.reshape(tag, c.shape)
    plt.figure(2)
    plt.imshow(tag)
    plt.axis('off')
    plt.show()
    # gmm
    res = gmm(data, 3, iter_times=20)
    tag = res[:, -1]
    tag = np.reshape(tag, c.shape)
    plt.figure(3)
    plt.imshow(tag)
    plt.axis('off')
    plt.show()
    # dbscan
    res = dbscan(data, 10, 100, dist_func=dist)
    tag = res[:, -1]
    tag = np.reshape(tag, c.shape)
    plt.figure(4)
    plt.imshow(tag)
    plt.axis('off')
    plt.show()

    # gray image
    img = cv2.imread('data/1.jpg', cv2.IMREAD_GRAYSCALE)
    plt.figure(1)
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    c = np.loadtxt('data/1.txt')
    plt.axis('off')
    plt.imshow(c)
    plt.show()
    pos = np.zeros((img.shape[0], img.shape[1], 2))
    for i in range(pos.shape[0]):
        for j in range(pos.shape[1]):
            pos[i, j, :] = [i, j]
    pos = pos.reshape((-1, 2))
    data = np.hstack((pos, np.reshape(img, (-1, 1))))
    # k_means
    res = k_means(data, 4, iter_times=100, dist_func=dist)
    tag = res[:, -1]
    tag = np.reshape(tag, c.shape)
    plt.figure(2)
    plt.imshow(tag)
    plt.axis('off')
    plt.show()
    # gmm
    res = gmm(data, 4, iter_times=50)
    tag = res[:, -1]
    tag = np.reshape(tag, c.shape)
    plt.figure(3)
    plt.imshow(tag)
    plt.axis('off')
    plt.show()
    # dbscan
    res = dbscan(data, 10, 70, dist_func=dist)
    tag = res[:, -1]
    tag = np.reshape(tag, c.shape)
    plt.figure(4)
    plt.imshow(tag)
    plt.axis('off')
    plt.show()
