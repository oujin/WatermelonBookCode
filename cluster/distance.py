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
    return minkowski_distance(x, y)


def minkowski_distance(x, y, p=2):
    """
    Parameters:
    ----------
    x: m-D array-like
        a cluster points, e.g. array([[x1,y1],...,[xn,yn]])
    y: 1-D array-like
        one point, e.g. array([x1,y1]).
    p: int or np.inf
        subscrpit of the L-p norm, p >= 1.
    Returns:
    ----------
    dist: 1-D array-like
        the minkowski distance from each one in x to y,
        e.g. array([d1,...,dn])
    """
    if p < 1:
        raise("p is too less than 1!")
    if p < np.inf:
        return (np.sum((x - y) ** p, -1)) ** (1/p)
    return np.max(np.abs((x - y)), -1)


def manhattan_distance(x, y):
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
        the manhattan distance from each one in x to y,
        e.g. array([d1,...,dn])
    """
    return minkowski_distance(x, y, p=1)


def chebyshev_distance(x, y):
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
        the chebyshev distance from each one in x to y,
        e.g. array([d1,...,dn])
    """
    return minkowski_distance(x, y, p=np.inf)


if __name__ == "__main__":
    x = np.array([
        [5, 2],
        [2, 2],
        [6, 1]
    ])
    y = np.array([1, 1])
    print(euclidean_distance(x, y))
    print(minkowski_distance(x, y, 2))
    print(manhattan_distance(x, y))
    print(minkowski_distance(x, y, 1))
    print(chebyshev_distance(x, y))
    print(minkowski_distance(x, y, np.inf))
