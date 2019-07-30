import numpy as np
from deprecated.kmeans import Kmeans
from deprecated.utils import Utils


class Particle:

    def __init__(self, n_cluster: int, data: np.ndarray, use_kmeans: bool = False, w: float = 0.9, c1: float = 0.5,
                 c2: float = 0.3):
        index = np.random.choice(list(range(len(data))), n_cluster)
        self.__centroids = data[index].copy()
        if use_kmeans:
            kmeans = Kmeans(n_cluster=n_cluster, init_pp=False)
            kmeans.fit(data)
            self.__centroids = kmeans.centroid.copy()
        self.__best_position = self.__centroids.copy()
        self.__best_score = Utils.quantization_error(self.__centroids, self.predict(data), data)
        self.__best_sse = Utils.calc_sse(self.__centroids, self.predict(data), data)
        self.__velocity = np.zeros_like(self.__centroids)
        self.__w = w
        self.__c1 = c1
        self.__c2 = c2

    def update(self, gbest_position: np.ndarray, data: np.ndarray):
        self.__update_velocity(gbest_position)
        self.__update_centroids(data)

    def predict(self, data: np.ndarray) -> np.ndarray:
        distance = self.__calc_distance(data)
        cluster = self.__assign_cluster(distance)
        return cluster

    def __update_velocity(self, gbest_position: np.ndarray):
        v_old = self.__w * self.__velocity
        cognitive_component = self.__c1 * np.random.random() * (self.__best_position - self.__centroids)
        social_component = self.__c2 * np.random.random() * (gbest_position - self.__centroids)
        self.velocity = v_old + cognitive_component + social_component

    def __update_centroids(self, data: np.ndarray):
        self.__centroids = self.__centroids + self.__velocity
        new_score = Utils.quantization_error(self.__centroids, self.predict(data), data)
        sse = Utils.calc_sse(self.__centroids, self.predict(data), data)
        self.__best_sse = min(sse, self.__best_sse)
        if new_score < self.__best_score:
            self.__best_score = new_score
            self.__best_position = self.__centroids.copy()

    def __calc_distance(self, data: np.ndarray) -> np.ndarray:
        distances = []
        for c in self.__centroids:
            distance = np.sum((data - c) * (data - c), axis=1)
            distances.append(distance)

        distances = np.array(distances)
        distances = np.transpose(distances)
        return distances

    @staticmethod
    def __assign_cluster(distance: np.ndarray) -> np.ndarray:
        cluster = np.argmin(distance, axis=1)
        return cluster

    @property
    def best_score(self):
        return self.__best_score

    @property
    def centroids(self):
        return self.__centroids

    @property
    def best_sse(self):
        return self.__best_sse
