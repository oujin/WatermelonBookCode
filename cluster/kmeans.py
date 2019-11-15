import matplotlib.pyplot as plt
import numpy as np

from distance import euclidean_distance
from estimate import Estimater


def k_means(data, k=2, iter_times=np.inf, dist_func=euclidean_distance):
    """
    Parameters:
    ----------
    data: m-D array-like
        e.g. array([[x1,y1],...,[xn,yn]])
    k: int
        the number of clusters, default 2.
    iter_times: int
        the iterative times of the algorithm, default infty,
        i.e. to stop if the mean vectors don't update anymore.
    dist_func: function
        a function to compute the distance with some measure, i.e.
        euclidean distance.
    Returns:
    ----------
    result: (m+1)-D array-like
        e.g. array([[x1,y1,c1],...,[xn,yn,cn]])
    """
    # count iterative times finished
    if data.shape[0] < k:
        raise Exception("The number of cluster is more than data.")
    times = 0
    # mean vectors
    mean_vectors = data[:k].copy()
    # mean vectors updated or not
    updated = True
    # the last column is cluster tag
    results = np.zeros((data.shape[0], data.shape[1] + 1))
    results[:, :-1] = data

    while times < iter_times and updated:
        for p in results:
            dist = dist_func(
                mean_vectors, p[:-1]
            )
            p[-1] = np.argmin(dist)
        # update mean vectors
        updated = False
        for i in range(k):
            new_mean = np.mean(results[results[:, -1] == i, :-1], 0)
            # if any mean vectors updated, iteration continues
            if (new_mean != mean_vectors[i]).any():
                mean_vectors[i] = new_mean
                updated = True
        times += 1
    return results


if __name__ == "__main__":
    data = np.random.normal(10, 9, (300, 2))
    data = np.vstack((data, np.random.normal(50, 7, (300, 2))))
    data = np.vstack((data, np.random.normal(35, 5, (300, 2))))
    tags = np.vstack((np.ones((300, 1)),
                      np.ones((300, 1)) * 2,
                      np.ones((300, 1)) * 3))
    index = np.arange(900)
    np.random.shuffle(index)
    results = k_means(data[index], k=5)
    for i in range(10):
        points = results[results[:, -1] == i]
        plt.scatter(points[:, 0], points[:, 1])
    plt.show()

    # estimate
    etr = Estimater()
    etr.get_internal_index(results, euclidean_distance)
    results = np.hstack((results, tags[index]))
    etr.get_external_index(results)
