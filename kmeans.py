import numpy

from utils import Utils


class Kmeans:

    def __init__(self, n_cluster: int, init_pp: bool = True, max_iter: int = 300, tolerance: float = 1e-4,
                 seed: int = None):
        self.__n_cluster = n_cluster
        self.__max_iter = max_iter
        self.__tolerance = tolerance
        self.__init_pp = init_pp
        self.__seed = seed
        self.__centroid = None
        self.__SSE = None

    @property
    def sse(self):
        return self.__SSE

    @sse.setter
    def sse(self, sse):
        self.__SSE = sse

    @property
    def centroid(self):
        return self.__centroid

    @centroid.setter
    def centroid(self, centroid):
        self.__centroid = centroid

    def _init_centroid(self, data: numpy.ndarray):
        if self.__init_pp:
            numpy.random.seed(self.__seed)
            centroid = [int(numpy.random.uniform() * len(data))]
            for _ in range(1, self.__n_cluster):
                dist = []
                dist = [min([numpy.inner(data[c] - x, data[c] - x) for c in centroid])
                        for i, x in enumerate(data)]
                dist = numpy.array(dist)
                dist = dist / dist.sum()
                cumdist = numpy.cumsum(dist)

                prob = numpy.random.rand()
                for i, c in enumerate(cumdist):
                    if prob > c and i not in centroid:
                        centroid.append(i)
                        break
            centroid = numpy.array([data[c] for c in centroid])
        else:
            numpy.random.seed(self.__seed)
            idx = numpy.random.choice(range(len(data)), size=self.__n_cluster)
            centroid = data[idx]
        return centroid

    def fit(self, data: numpy.ndarray):
        self.__centroid = self._init_centroid(data)
        for _ in range(self.__max_iter):
            distance = self.__calc_distance(data)
            cluster = self._assign_cluster(distance)
            new_centroid = self._update_centroid(data, cluster)
            diff = numpy.abs(self.__centroid - new_centroid).mean()
            self.__centroid = new_centroid

            if diff <= self.__tolerance:
                break

        self.__SSE = Utils.calc_sse(self.__centroid, cluster, data)

    def predict(self, data: numpy.ndarray):
        distance = self.__calc_distance(data)
        cluster = self._assign_cluster(distance)
        return cluster

    @staticmethod
    def _assign_cluster(distance: numpy.ndarray):
        cluster = numpy.argmin(distance, axis=1)
        return cluster

    def _update_centroid(self, data: numpy.ndarray, cluster: numpy.ndarray):
        centroids = []
        for i in range(self.__n_cluster):
            idx = numpy.where(cluster == i)
            centroid = numpy.mean(data[idx], axis=0)
            centroids.append(centroid)
        centroids = numpy.array(centroids)
        return centroids

    def __calc_distance(self, data: numpy.ndarray):
        distances = []
        for c in self.__centroid:
            distance = numpy.sum((data - c) * (data - c), axis=1)
            distances.append(distance)

        distances = numpy.array(distances)
        distances = distances.T
        return distances
