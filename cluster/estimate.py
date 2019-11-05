import numpy as np


class Estimater(object):
    """
    An estimater to estimate cluster algorithms.
    """

    def get_external_index(self, data):
        """
        Parameters:
        ----------
        data: m-D array-like
            e.g. array([[x1,y1,c1,c1*],...,[xn,yn,cn,cn*]])
        Returns:
        ----------
        """
        # a: the number of pairs that in the same C and the same C*
        # b: the number of pairs that in the same C but not the same C*
        # c: the number of pairs that in the same C* but not the same C
        # d: the number of pairs that neither in the same C nor C*
        a, b, c, d = 0, 0, 0, 0
        for i in range(data.shape[0]):
            for j in range(i+1, data.shape[0]):
                if data[i, -1] == data[j, -1] and data[i, -2] == data[j, -2]:
                    a += 1
                elif data[i, -2] == data[j, -2]:
                    b += 1
                elif data[i, -1] == data[j, -1]:
                    c += 1
                else:
                    d += 1
        # Jaccard Coefficient
        jc = a / (a + b + c)
        # Fowlkes and Mallows Index
        fmi = np.sqrt((a ** 2) / ((a + b) * (a + c)))
        # Rand Index
        ri = (a + d) / (a + b + c + d)
        print()
        print("{:20s}|{:25s}|{:10s}".format("Jaccard Coefficient",
                                            "Fowlkes and Mallows Index",
                                            "Rand Index")
              )
        print("{:20s}|{:25s}|{:10s}".format("-" * 20, "-" * 25, "-" * 10))
        print("{:20.6f}|{:25.6f}|{:10.6f}".format(jc, fmi, ri))

    def _get_avg_and_diam(self, cluster, dist_func):
        """
        Parameters:
        ----------
        cluster: m-D array-like
            e.g. array([[x1,y1,c1],...,[xn,yn,c1]]), all points in a same
            cluster.
        dist_func: function
            a function to compute the distance with some measure, i.e.
            euclidean distance.
        Returns:
        ----------
        avg: float
            the mean distance between points in the same cluster.
        diam: float
            the max distance between points in the same cluster.
        """
        dists = []
        for i in range(cluster.shape[0]):
            for j in range(i+1, cluster.shape[0]):
                dists.append(dist_func(cluster[i, :-1], cluster[j, :-1]))
        avg = np.mean(dists)
        diam = np.max(dists)
        return avg, diam

    def _get_dmin(self, cluster1, cluster2, dist_func):
        """
        Parameters:
        ----------
        cluster1: m-D array-like
            e.g. array([[x1,y1,c1],...,[xn,yn,c1]]), all points in a same
            cluster.
        cluster2: m-D array-like
            e.g. array([[x1,y1,c2],...,[xn,yn,c2]]), all points in a same
            cluster.
        dist_func: function
            a function to compute the distance with some measure, i.e.
            euclidean distance.
        Returns:
        ----------
        dmin: float
            the min distance between the two clusters.
        """
        dists = []
        for i in range(cluster1.shape[0]):
            for j in range(cluster2.shape[0]):
                dists.append(dist_func(cluster1[i, :-1], cluster2[j, :-1]))
        dmin = np.min(dists)
        return dmin

    def get_internal_index(self, data, dist_func):
        """
        Parameters:
        ----------
        data: m-D array-like
            e.g. array([[x1,y1,c1],...,[xn,yn,cn]]).
        dist_func: function
            a function to compute the distance with some measure, i.e.
            euclidean distance.
        Returns:
        ----------
        """
        num_clusters = np.unique(data[:, -1]).shape[0]
        # centers: the centers of cluster
        # avgs: the mean distance between points in the same cluster
        centers, avgs = [], []
        # diam: the max distance between points in the same cluster
        max_diam = 0
        for i in range(num_clusters):
            avg, diam = self._get_avg_and_diam(data[data[:, -1] == i],
                                               dist_func)
            max_diam = max(max_diam, diam)
            avgs.append(avg)
            centers.append(np.mean(data[data[:, -1] == i, :-1], 0))

        avgs_over_dcen = np.zeros((num_clusters, num_clusters))
        # dmin: the min distance between some two clusters
        dmins = np.ones((num_clusters, num_clusters)) * np.inf
        for i in range(num_clusters):
            for j in range(i+1, num_clusters):
                # dcen: distance between the center points of clusters
                dcen = dist_func(centers[i], centers[j])
                avgs_over_dcen[j, i] = (avgs[i] + avgs[j]) / dcen
                avgs_over_dcen[i, j] = avgs_over_dcen[j, i]
                dmins[i, j] = dmins[j, i] = self._get_dmin(
                    data[data[:, -1] == i], data[data[:, -1] == j], dist_func)

        # Davies-Boudin Index
        dbi = np.mean(np.max(avgs_over_dcen, 0))
        # Dunn Index
        di = np.min(dmins) / max_diam
        print()
        print("{:20s}|{:10s}".format("Davies-Boudin Index", "Dunn Index"))
        print("{:20s}|{:10s}".format("-" * 20, "-" * 10))
        print("{:20.6f}|{:10.6f}".format(dbi, di))
