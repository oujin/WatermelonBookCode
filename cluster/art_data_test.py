import matplotlib.pyplot as plt
import numpy as np

from distance import euclidean_distance, manhattan_distance
from distance import chebyshev_distance
from estimate import Estimater
from kmeans import k_means
from gmm import gmm
from dbscan import dbscan


if __name__ == "__main__":
    # show data
    for k in range(1, 4):
        data = np.load("data/{}.npy".format(k))
        plt.subplot(1, 3, k)
        for i in range(10):
            points = data[data[:, -1] == i]
            plt.scatter(points[:, 0], points[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("data{} distributions".format(k))
    plt.show()

    # Experiment1：data and models
    # kmeans
    for k in range(1, 4):
        data = np.load("data/{}.npy".format(k))
        results = k_means(data[:, :-1], k=3)
        plt.subplot(1, 3, k)
        for i in range(10):
            points = results[results[:, -1] == i]
            plt.scatter(points[:, 0], points[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(
            "k-means clusters(with euclidean distance) over data{}".format(k)
        )
        print("method: kmeans - data{} - euclidean distance".format(k))
        etr = Estimater()
        etr.get_internal_index(results, euclidean_distance)
        results = np.hstack((results, data[:, -1:]))
        etr.get_external_index(results)
    plt.show()

    # gmm
    for k in range(1, 4):
        data = np.load("data/{}.npy".format(k))
        results = gmm(data[:, :-1], k=3)
        plt.subplot(1, 3, k)
        for i in range(10):
            points = results[results[:, -1] == i]
            plt.scatter(points[:, 0], points[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("gmm clusters over data{}".format(k))
        print("method: gmm - data{}".format(k))
        etr = Estimater()
        etr.get_internal_index(results, euclidean_distance)
        results = np.hstack((results, data[:, -1:]))
        etr.get_external_index(results)
    plt.show()

    # dbscan
    for k in range(1, 4):
        data = np.load("data/{}.npy".format(k))
        results = dbscan(data[:, :-1], 3, 20)
        plt.subplot(1, 3, k)
        for i in range(1, 10):
            points = results[results[:, -1] == i]
            plt.scatter(points[:, 0], points[:, 1])
        noise = results[results[:, -1] == 0]
        plt.scatter(noise[:, 0], noise[:, 1], marker='^', label="noise")
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(
            "dbscan clusters(with euclidean distance) over data{}".format(k)
        )
        print("method: dbscan - data{} - euclidean distance".format(k))
        etr = Estimater()
        etr.get_internal_index(results, euclidean_distance)
        results = np.hstack((results, data[:, -1:]))
        etr.get_external_index(results)
    plt.show()

    # Experiment2：distance metrics
    # different distance
    dist = [('manhattan distance', manhattan_distance),
            ('chebyshev distance', chebyshev_distance)
            ]
    # kmeans
    for d in dist:
        for k in range(1, 4):
            data = np.load("data/{}.npy".format(k))
            results = k_means(data[:, :-1], k=3, dist_func=d[1])
            plt.subplot(1, 3, k)
            for i in range(10):
                points = results[results[:, -1] == i]
                plt.scatter(points[:, 0], points[:, 1])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(
                "k-means clusters(with {}) over data{}".format(d[0], k)
            )
            print("method: kmeans - data{} - {}".format(k, d[0]))
            etr = Estimater()
            etr.get_internal_index(results)
            results = np.hstack((results, data[:, -1:]))
            etr.get_external_index(results)
        plt.show()

    # dbscan
    for d in dist:
        for k in range(1, 4):
            data = np.load("data/{}.npy".format(k))
            results = dbscan(data[:, :-1], 3, 20, dist_func=d[1])
            plt.subplot(1, 3, k)
            for i in range(1, 20):
                points = results[results[:, -1] == i]
                plt.scatter(points[:, 0], points[:, 1])
            noise = results[results[:, -1] == 0]
            plt.scatter(noise[:, 0], noise[:, 1], marker='^', label="noise")
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title("dbscan clusters(with {}) over data{}".format(d[0], k))
            print("method: dbscan - data{} - {}".format(k, d[0]))
            etr = Estimater()
            etr.get_internal_index(results)
            results = np.hstack((results, data[:, -1:]))
            etr.get_external_index(results)
        plt.show()

    # Experiment3：outlier
    # show data with outliers
    data = [np.load("data/1.npy"),
            np.load("data/2.npy"),
            np.load("data/3.npy")]
    noise_p = np.ones((10, 3))
    noise_p[:, :-1] = np.random.uniform([-450, 20], [-400, 60], (10, 2))
    data[0] = np.vstack((data[0], np.array(noise_p)))
    data[1] = np.vstack((data[1], np.array(noise_p)))
    data[2] = np.vstack((data[2], np.array(noise_p)))
    for k in range(3):
        plt.subplot(1, 3, k+1)
        dat, outlier = data[k][:-10, :], data[k][-10:, :]
        for i in range(10):
            points = dat[dat[:, -1] == i]
            if i == outlier[-1, -1]:
                plt.scatter(points[:, 0], points[:, 1], color='b')
            else:
                plt.scatter(points[:, 0], points[:, 1])
        plt.scatter(outlier[-10:, 0], outlier[-10:, 1], c="b",
                    marker='^', label="outlier")
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-460, 80)
        plt.ylim(-20, 80)
        plt.title("data{} distributions".format(k+1))
    plt.show()

    # kmeans
    for k in range(3):
        plt.subplot(1, 3, k+1)
        results = k_means(data[k][:, :-1], k=3)
        for i in range(10):
            points = results[results[:, -1] == i]
            plt.scatter(points[:, 0], points[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-460, 80)
        plt.ylim(-20, 80)
        plt.title(
            "k-means clusters(with euclidean distance) over data{}".format(k+1)
        )
    plt.show()

    # gmm
    for k in range(3):
        plt.subplot(1, 3, k+1)
        results = gmm(data[k][:, :-1], k=3)
        for i in range(10):
            points = results[results[:, -1] == i]
            plt.scatter(points[:, 0], points[:, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-460, 80)
        plt.ylim(-20, 80)
        plt.title(
            "gmm clusters over data{}".format(k+1)
        )
    plt.show()

    # dbscan
    for k in range(3):
        plt.subplot(1, 3, k+1)
        results = dbscan(data[k][:, :-1], 3, 20)
        for i in range(1, 20):
            points = results[results[:, -1] == i]
            plt.scatter(points[:, 0], points[:, 1])
        noise = results[results[:, -1] == 0]
        plt.scatter(noise[:, 0], noise[:, 1], marker='^', label="noise")
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-460, 80)
        plt.ylim(-20, 80)
        plt.title(
            "dbscan clusters(with euclidean distance) over data{}".format(k+1)
        )
    plt.show()

    # Experiment4：noise
    data0 = np.load("data/3.npy")
    # dbscan
    res = [[], []]
    for k in range(0, 4):
        noise_n = int(data0.shape[0] * 0.1 * k)
        noise_p = np.ones((noise_n, 3))
        noise_p[:, :-1] = np.random.uniform([0, 0], [70, 70], (noise_n, 2))
        data = np.vstack((data0, np.array(noise_p)))
        plt.subplot(2, 2, k+1)
        res[0].append(dbscan(data[:, :-1], 3, 20))
        res[1].append(dbscan(data[:, :-1], 3, 30))
    plt.figure(1)
    for k in range(0, 4):
        results = res[0][k]
        plt.subplot(2, 2, k+1)
        for i in range(1, 20):
            points = results[results[:, -1] == i]
            plt.scatter(points[:, 0], points[:, 1])
        noise = results[results[:, -1] == 0]
        plt.scatter(noise[:, 0], noise[:, 1], marker='^', label="noise")
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(
            "anti-noise test(N/S:{:3.2f}) - dbscan clusters".format(0.1 * k)
        )
    # plt.show()
    plt.figure(2)
    for k in range(0, 4):
        results = res[1][k]
        plt.subplot(2, 2, k+1)
        for i in range(1, 20):
            points = results[results[:, -1] == i]
            plt.scatter(points[:, 0], points[:, 1])
        noise = results[results[:, -1] == 0]
        plt.scatter(noise[:, 0], noise[:, 1], marker='^', label="noise")
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(
            "anti-noise test(N/S:{:3.2f}) - dbscan clusters".format(0.1 * k)
        )
    plt.show()
