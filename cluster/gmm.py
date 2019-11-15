import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg

from distance import euclidean_distance
from estimate import Estimater


def _gaussian_pdf(mu, sigma):
    """
    Parameters:
    ----------
    mu: n-D array-like or float
        e.g. array([mu1, ..., mun])
    sigma: n-D array-like or float
        e.g. array([sigma1, ..., sigman])
    Returns:
    ----------
    pdf_func: function
        a pdf function of gaussian with mu and sigma.
    """
    if isinstance(mu, int) or isinstance(mu, float):
        mu = np.asarray([mu])
    if isinstance(sigma, int) or isinstance(sigma, float):
        sigma = np.asarray([[sigma]])
    # avoid singular
    inv_sigma = lg.inv(sigma + np.eye(sigma.shape[0]) * 1e-10)
    # avoid trunc
    det_sigma = max(lg.det(sigma), 1e-20)
    b = ((2 * np.pi) ** (mu.shape[0] / 2)) * np.sqrt(det_sigma)

    def pdf_func(x):
        if x.shape != mu.shape:
            raise Exception("Please check the shape of x and mu.")
        a = np.dot(np.dot((x - mu).T, inv_sigma), (x - mu))
        return np.exp(0 - a / 2) / b
    return pdf_func


def gmm(data, k=2, iter_times=np.inf, epsilon=1e-2):
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
    epsilon: float
        if the likelihood is less than epsilon, the iteration stops.
    Returns:
    ----------
    result: (m+1)-D array-like
        e.g. array([[x1,y1,c1],...,[xn,yn,cn]])
    """
    # count iterative times finished
    if data.shape[0] < k:
        raise Exception("The number of cluster is more than data.")
    times = 0
    # initialize alpha
    alpha = np.ones(k) / k
    # initialize mus
    mus = [data[i].copy() for i in range(k)]
    # initialize sigmas
    sigmas = [np.eye(data.shape[1]) * 10 for i in range(k)]
    # the last column is cluster tag
    results = np.zeros((data.shape[0], data.shape[1] + 1))
    results[:, :-1] = data
    # probability matrix
    pr = np.zeros((data.shape[0], k))
    # likelihood
    likelihood = -np.inf

    while times < iter_times:
        # pdf functions
        pdfs = []
        for mu, sigma in zip(mus, sigmas):
            pdfs.append(_gaussian_pdf(mu, sigma))
        # new likelihood
        new_lld = 0
        # update probability matrix
        for i, p in enumerate(data):
            pr[i, :] = [a * pdf(p) for a, pdf in zip(alpha, pdfs)]
            sum_pr = max(np.sum(pr[i, :]), 1e-10)
            new_lld += np.log(sum_pr)
            pr[i, :] = pr[i, :] / sum_pr
        for i in range(k):
            # new parameters
            mus[i] = np.dot(pr[:, i].T, data) / max(np.sum(pr[:, i]), 1e-10)
            simga = np.zeros((data.shape[1], data.shape[1]))
            for j, d in enumerate(data):
                delta = np.expand_dims(d - mus[i], axis=1)
                simga += (np.dot(delta, delta.T) * pr[j, i])
            sigmas[i] = simga / max(np.sum(pr[:, i]), 1e-10)
            # sigmas[i] = np.dot(np.dot((data - mus[i]).T, np.diag(pr[:, i])),
            #                    data - mus[i]) / max(np.sum(pr[:, i]), 1e-10)
            alpha[i] = max(np.mean(pr[:, i]), 1e-10)
        times += 1
        if np.abs(likelihood - new_lld) < epsilon:
            break
        likelihood = new_lld
    # clusters
    pdfs = []
    for mu, sigma in zip(mus, sigmas):
        pdfs.append(_gaussian_pdf(mu, sigma))
    # probability matrix
    pr = np.zeros((data.shape[0], k))
    for i, p in enumerate(data):
        pr[i, :] = [a * pdf(p) for a, pdf in zip(alpha, pdfs)]
    results[:, -1] = np.argmax(pr, 1)
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
    results = gmm(data[index], k=3)
    for i in range(10):
        points = results[results[:, -1] == i]
        plt.scatter(points[:, 0], points[:, 1])

    # estimate
    etr = Estimater()
    etr.get_internal_index(results, euclidean_distance)
    results = np.hstack((results, tags[index]))
    etr.get_external_index(results)

    plt.show()
