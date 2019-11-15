import matplotlib.pyplot as plt
import numpy as np

from distance import euclidean_distance
from estimate import Estimater


def dbscan(data, epsilon=1., min_pts=1, dist_func=euclidean_distance):
    """
    Parameters:
    ----------
    data: m-D array-like
        e.g. array([[x1,y1],...,[xn,yn]])
    epsilon: float
        define a neighbourhood, default 1.0.
    min_pts: int
        only if the neighbourhood of x contains min_pts points
        at least, x is called a core point.
    dist_func: function
        a function to compute the distance with some measure, i.e.
        euclidean distance.
    Returns:
    ----------
    result: (m+1)-D array-like
        e.g. array([[x1,y1,c1],...,[xn,yn,cn]])
    """
    # count iterative times finished
    if data.shape[0] < min_pts:
        return np.empty((0, data.shape[1]))
    # the last column is cluster tag
    results = np.zeros((data.shape[0], data.shape[1] + 1))
    results[:, :-1] = data
    # core points
    cores = []
    for i, p in enumerate(data):
        dist = dist_func(data, p)
        if np.sum(dist < epsilon) > min_pts:
            cores.append(i)
    cores = np.asarray(cores)
    # clusters number
    k = 0
    unvisited = np.arange(data.shape[0])
    while len(cores) > 0:
        unvisited_old = unvisited.copy()
        Q = [np.random.choice(cores)]
        unvisited = np.setdiff1d(unvisited, Q)
        while len(Q) > 0:
            # pop q
            q, Q = Q[0], Q[1:]
            if q in cores:
                dist = dist_func(data, data[int(q), :])
                Delta = np.intersect1d(np.where(dist < epsilon), unvisited)
                Q = np.union1d(Q, Delta)
                unvisited = np.setdiff1d(unvisited, Delta)
        k += 1
        cluster = np.setdiff1d(unvisited_old, unvisited)
        results[cluster, -1] = k
        cores = np.setdiff1d(cores, cluster)
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
    results = dbscan(data[index], epsilon=5, min_pts=30)
    for i in range(10):
        points = results[results[:, -1] == i]
        plt.scatter(points[:, 0], points[:, 1])
    plt.show()

    # estimate
    etr = Estimater()
    etr.get_internal_index(results, euclidean_distance)
    results = np.hstack((results, tags[index]))
    etr.get_external_index(results)
