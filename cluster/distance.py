import numpy as np


def euclidean_distance(x, y):
    """
    Parameters:
    ----------
    x: m-D array-like
        a cluster points, e.g. array([[x1,y1],...,[xn,yn]])
    y: 1-D array-like
        one point, e.g. array([x1,y1]).
    Returns:
    ----------
    dist: 1-D array-like
        the euclidean distance from each one in x to y,
        e.g. array([d1,...,dn])
    """
    return np.sqrt(np.sum((x - y) ** 2, 1))


if __name__ == "__main__":
    x = np.array([
        [5, 2],
        [2, 2],
        [6, 1]
    ])
    y = np.array([1, 1])
    print(euclidean_distance(x, y))
