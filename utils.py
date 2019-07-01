import numpy as np


class Utils:

    def __init__(self):
        pass

    @staticmethod
    def calc_sse(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray):
        distances = 0
        for i, c in enumerate(centroids):
            idx = np.where(labels == i)
            dist = np.sum((data[idx] - c) ** 2)
            distances += dist
        return distances

    @staticmethod
    def quantization_error(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray) -> float:
        error = 0.0
        for i, c in enumerate(centroids):
            idx = np.where(labels == i)
            dist = np.linalg.norm(data[idx] - c)
            dist /= len(idx)
            error += dist
        error /= len(centroids)
        return error
