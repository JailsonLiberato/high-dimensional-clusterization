import numpy as np
from utils import Utils


class Kmeans:

    def __init__(self, n_cluster: int, init_pp:bool = True, max_iter: int = 300, tolerance: float = 1e-4):
        self.__n_cluster = n_cluster
        self.__init_pp = init_pp
        self.__max_iter = max_iter
        self.__tolerance = tolerance
        self.__centroid = None
        self.__sse = None

    def fit(self, data: np.ndarray):
        self.__centroid = self.__init_centroid(data)
        for _ in range(self.__max_iter):
            distance = self.__calc_distance(data)
            cluster = self.__assign_cluster(distance)
            new_centroid = self.__update_centroid(data, cluster)
            diff = np.abs(self.__centroid - new_centroid).mean()
            self.__centroid = new_centroid

            if diff <= self.__tolerance:
                break

        self.__sse = Utils.calc_sse(self.__centroid, cluster, data)

    def predict(self, data: np.ndarray):
        distance = self.__calc_distance(data)
        cluster = self.__assign_cluster(distance)
        return cluster

    def __calc_distance(self, data: np.ndarray):
        distances = []
        for c in self.__centroid:
            distance = np.sum((data - c) * (data - c), axis=1)
            distances.append(distance)
        distances = np.array(distances)
        distances = distances.T
        return distances

    def __update_centroid(self, data: np.ndarray, cluster: np.ndarray):
        centroids = []
        for i in range(self.__n_cluster):
            idx = np.where(cluster == i)
            centroid = np.mean(data[idx], axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        return centroids

    def __init_centroid(self, data: np.ndarray):
        if self.__init_pp:
            centroid = [int(np.random.uniform() * len(data))]
            for _ in range(1, self.__n_cluster):
                dist = [min([np.inner(data[c] - x, data[c] - x) for c in centroid])
                        for i, x in enumerate(data)]
                dist = np.array(dist)
                dist = dist / dist.sum()
                cumdist = np.cumsum(dist)

                prob = np.random.rand()
                for i, c in enumerate(cumdist):
                    if prob > c and i not in centroid:
                        centroid.append(i)
                        break
            centroid = np.array([data[c] for c in centroid])
        else:
            idx = np.random.choice(range(len(data)), size=self.__n_cluster)
            centroid = data[idx]
        return centroid

    @staticmethod
    def __assign_cluster(distance: np.ndarray):
        cluster = np.argmin(distance, axis=1)
        return cluster

    @property
    def sse(self):
        return self.__sse

    @property
    def centroid(self):
        return self.__centroid

    @centroid.setter
    def centroid(self, centroid):
        self.__centroid = centroid


